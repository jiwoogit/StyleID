import argparse
import glob
import numpy as np
import os
from PIL import Image
from scipy import linalg
import torch
from sklearn.linear_model import LinearRegression
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, Grayscale
from tqdm import tqdm

import utils
import inception
import image_metrics

ALLOWED_IMAGE_EXTENSIONS = ['jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG']
CKPT_URL = 'https://huggingface.co/matthias-wright/art_inception/resolve/main/art_inception.pth'


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img


def get_activations(files, model, batch_size=50, device='cpu', num_workers=1):
    """Computes the activations of for all images.

    Args:
        files (list): List of image file paths.
        model (torch.nn.Module): Model for computing activations.
        batch_size (int): Batch size for computing activations.
        device (torch.device): Device for commputing activations.
        num_workers (int): Number of threads for data loading.

    Returns:
        (): Activations of the images, shape [num_images, 2048].
    """
    model.eval()

    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)

    dataset = ImagePathDataset(files, transforms=Compose([Resize(512),ToTensor()]))
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)

    pred_arr = np.empty((len(files), 2048))

    start_idx = 0

    pbar = tqdm(total=len(files))
    for batch in dataloader:
        batch = batch.to(device)

        with torch.no_grad():
            features = model(batch, return_features=True)

        features = features.cpu().numpy()
        pred_arr[start_idx:start_idx + features.shape[0]] = features
        start_idx = start_idx + features.shape[0]

        pbar.update(batch.shape[0])

    pbar.close()
    return pred_arr


def compute_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    
    Args:
        mu1 (np.ndarray): Sample mean of activations of stylized images.
        mu2 (np.ndarray): Sample mean of activations of style images.
        sigma1 (np.ndarray): Covariance matrix of activations of stylized images.
        sigma2 (np.ndarray): Covariance matrix of activations of style images.
        eps (float): Epsilon for numerical stability.

    Returns:
        (float) FID value.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def compute_activation_statistics(files, model, batch_size=50, device='cpu', num_workers=1):
    """Computes the activation statistics used by the FID.
    
    Args:
        files (list): List of image file paths.
        model (torch.nn.Module): Model for computing activations.
        batch_size (int): Batch size for computing activations.
        device (torch.device): Device for commputing activations.
        num_workers (int): Number of threads for data loading.

    Returns:
        (np.ndarray, np.ndarray): mean of activations, covariance of activations
        
    """
    act = get_activations(files, model, batch_size, device, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def get_image_paths(path, sort=False):
    """Returns the paths of the images in the specified directory, filtered by allowed file extensions.

    Args:
        path (str): Path to image directory.
        sort (bool): Sort paths alphanumerically.

    Returns:
        (list): List of image paths with allowed file extensions.

    """
    paths = []
    for extension in ALLOWED_IMAGE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(path, f'*.{extension}')))
    if sort:
        paths.sort()
    return paths

def compute_fid(path_to_stylized, path_to_style, batch_size, device, num_workers=1):
    """Computes the FID for the given paths.

    Args:
        path_to_stylized (str): Path to the stylized images.
        path_to_style (str): Path to the style images.
        batch_size (int): Batch size for computing activations.
        device (str): Device for commputing activations.
        num_workers (int): Number of threads for data loading.

    Returns:
        (float) FID value.
    """
    device = torch.device('cuda') if device == 'cuda' and torch.cuda.is_available() else torch.device('cpu')

    ckpt_file = utils.download(CKPT_URL)
    ckpt = torch.load(ckpt_file, map_location=device)
    
    model = inception.Inception3().to(device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    
    stylized_image_paths = get_image_paths(path_to_stylized)
    style_image_paths = get_image_paths(path_to_style)

    mu1, sigma1 = compute_activation_statistics(stylized_image_paths, model, batch_size, device, num_workers)
    mu2, sigma2 = compute_activation_statistics(style_image_paths, model, batch_size, device, num_workers)
    
    fid_value = compute_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid_value


def compute_fid_infinity(path_to_stylized, path_to_style, batch_size, device, num_points=15, num_workers=1):
    """Computes the FID infinity for the given paths.

    Args:
        path_to_stylized (str): Path to the stylized images.
        path_to_style (str): Path to the style images.
        batch_size (int): Batch size for computing activations.
        device (str): Device for commputing activations.
        num_points (int): Number of FID_N we evaluate to fit a line.
        num_workers (int): Number of threads for data loading.

    Returns:
        (float) FID infinity value.
    """
    device = torch.device('cuda') if device == 'cuda' and torch.cuda.is_available() else torch.device('cpu')

    ckpt_file = utils.download(CKPT_URL)
    ckpt = torch.load(ckpt_file, map_location=device)
    
    model = inception.Inception3().to(device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    stylized_image_paths = get_image_paths(path_to_stylized)
    style_image_paths = get_image_paths(path_to_style)

    assert len(stylized_image_paths) == len(style_image_paths), \
           f'Number of stylized images and number of style images must be equal.({len(stylized_image_paths)},{len(style_image_paths)})'

    activations_stylized = get_activations(stylized_image_paths, model, batch_size, device, num_workers)
    activations_style = get_activations(style_image_paths, model, batch_size, device, num_workers)
    activation_idcs = np.arange(activations_stylized.shape[0])

    fids = []
    
    fid_batches = np.linspace(start=5000, stop=len(stylized_image_paths), num=num_points).astype('int32')
    
    for fid_batch_size in fid_batches:
        np.random.shuffle(activation_idcs)
        idcs = activation_idcs[:fid_batch_size]
        
        act_style_batch = activations_style[idcs]
        act_stylized_batch = activations_stylized[idcs]

        mu_style, sigma_style = np.mean(act_style_batch, axis=0), np.cov(act_style_batch, rowvar=False)
        mu_stylized, sigma_stylized = np.mean(act_stylized_batch, axis=0), np.cov(act_stylized_batch, rowvar=False)
        
        fid_value = compute_frechet_distance(mu_style, sigma_style, mu_stylized, sigma_stylized)
        fids.append(fid_value)

    fids = np.array(fids).reshape(-1, 1)
    reg = LinearRegression().fit(1 / fid_batches.reshape(-1, 1), fids)
    fid_infinity = reg.predict(np.array([[0]]))[0,0]

    return fid_infinity


def compute_content_distance(path_to_stylized, path_to_content, batch_size, content_metric='lpips', device='cuda', num_workers=1, gray=False):
    """Computes the distance for the given paths.

    Args:
        path_to_stylized (str): Path to the stylized images.
        path_to_style (str): Path to the style images.
        batch_size (int): Batch size for computing activations.
        content_metric (str): Metric to use for content distance. Choices: 'lpips', 'vgg', 'alexnet'
        device (str): Device for commputing activations.
        num_workers (int): Number of threads for data loading.

    Returns:
        (float) FID value.
    """
    device = torch.device('cuda') if device == 'cuda' and torch.cuda.is_available() else torch.device('cpu')

    # Sort paths in order to match up the stylized images with the corresponding content image
    stylized_image_paths = get_image_paths(path_to_stylized, sort=True)
    content_image_paths = get_image_paths(path_to_content, sort=True)

    assert len(stylized_image_paths) == len(content_image_paths), \
           'Number of stylized images and number of content images must be equal.'

    if gray:
        content_transforms = Compose([Resize(512), Grayscale(),
        ToTensor()])
    else:
        content_transforms = Compose([Resize(512),
        ToTensor()])
    
    dataset_stylized = ImagePathDataset(stylized_image_paths, transforms=content_transforms)
    dataloader_stylized = torch.utils.data.DataLoader(dataset_stylized,
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      drop_last=False,
                                                      num_workers=num_workers)

    dataset_content = ImagePathDataset(content_image_paths, transforms=content_transforms)
    dataloader_content = torch.utils.data.DataLoader(dataset_content,
                                                     batch_size=batch_size,
                                                     shuffle=False,
                                                     drop_last=False,
                                                     num_workers=num_workers)
    
    metric_list = ['alexnet', 'ssim', 'ms-ssim']
    if content_metric in metric_list:
        metric = image_metrics.Metric(content_metric).to(device)
    elif content_metric == 'lpips':
        metric = image_metrics.LPIPS().to(device)
    elif content_metric == 'vgg':
        metric = image_metrics.LPIPS_vgg().to(device)
    else:
        raise ValueError(f'Invalid content metric: {content_metric}')

    dist_sum = 0.0
    N = 0
    pbar = tqdm(total=len(stylized_image_paths))
    for batch_stylized, batch_content in zip(dataloader_stylized, dataloader_content):
        with torch.no_grad():
            batch_dist = metric(batch_stylized.to(device), batch_content.to(device))
            N += batch_stylized.shape[0]
            dist_sum += torch.sum(batch_dist)

        pbar.update(batch_stylized.shape[0])

    pbar.close()

    return dist_sum / N

def compute_patch_simi(path_to_stylized, path_to_content, batch_size, device, num_workers=1):
    """Computes the distance for the given paths.

    Args:
        path_to_stylized (str): Path to the stylized images.
        path_to_style (str): Path to the style images.
        batch_size (int): Batch size for computing activations.
        content_metric (str): Metric to use for content distance. Choices: 'lpips', 'vgg', 'alexnet'
        device (str): Device for commputing activations.
        num_workers (int): Number of threads for data loading.

    Returns:
        (float) FID value.
    """
    device = torch.device('cuda') if device == 'cuda' and torch.cuda.is_available() else torch.device('cpu')

    # Sort paths in order to match up the stylized images with the corresponding content image
    stylized_image_paths = get_image_paths(path_to_stylized, sort=True)
    content_image_paths = get_image_paths(path_to_content, sort=True)

    assert len(stylized_image_paths) == len(content_image_paths), \
           'Number of stylized images and number of content images must be equal.'

    style_transforms = ToTensor()
    
    dataset_stylized = ImagePathDataset(stylized_image_paths, transforms=style_transforms)
    dataloader_stylized = torch.utils.data.DataLoader(dataset_stylized,
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      drop_last=False,
                                                      num_workers=num_workers)

    dataset_content = ImagePathDataset(content_image_paths, transforms=style_transforms)
    dataloader_content = torch.utils.data.DataLoader(dataset_content,
                                                     batch_size=batch_size,
                                                     shuffle=False,
                                                     drop_last=False,
                                                     num_workers=num_workers)
    
    metric = image_metrics.PatchSimi(device=device).to(device)

    dist_sum = 0.0
    N = 0
    pbar = tqdm(total=len(stylized_image_paths))
    for batch_stylized, batch_content in zip(dataloader_stylized, dataloader_content):
        with torch.no_grad():
            batch_dist = metric(batch_stylized.to(device), batch_content.to(device))
            N += batch_stylized.shape[0]
            dist_sum += torch.sum(batch_dist)

        pbar.update(batch_stylized.shape[0])

    pbar.close()

    return dist_sum / N

def compute_art_fid(path_to_stylized, path_to_style, path_to_content, batch_size, device, mode='art_fid_inf', content_metric='lpips', num_workers=1):
    """Computes the FID for the given paths.

    Args:
        path_to_stylized (str): Path to the stylized images.
        path_to_style (str): Path to the style images.
        path_to_content (str): Path to the content images.
        batch_size (int): Batch size for computing activations.
        device (str): Device for commputing activations.
        content_metric (str): Metric to use for content distance. Choices: 'lpips', 'vgg', 'alexnet'
        num_workers (int): Number of threads for data loading.

    Returns:
        (float) ArtFID value.
    """
    print('Compute FID value...')
    if mode == 'art_fid_inf':
        fid_value = compute_fid_infinity(path_to_stylized, path_to_style, batch_size, device, num_workers)
    elif mode == 'art_fid':
        fid_value = compute_fid(path_to_stylized, path_to_style, batch_size, device, num_workers)
    elif mode == 'style_loss':
        fid_value = compute_style_loss(path_to_stylized, path_to_style, batch_size, device, num_workers)
    else:
        fid_value = compute_gram_loss(path_to_stylized, path_to_style, batch_size, device, num_workers)
    
    print('Compute content distance...')
    cnt_value = compute_content_distance(path_to_stylized, path_to_content, batch_size, content_metric, device, num_workers)
    gray_cnt_value = compute_content_distance(path_to_stylized, path_to_content, batch_size, content_metric, device, num_workers, gray=True)

    art_fid_value = (cnt_value + 1) * (fid_value + 1)
    # fid_value = f'{fid_value.item():.4f}'
    # cnt_value = f'{content_dist.item():.4f}'
    # gray_cnt_value = f'{gray_content_dist.item():.4f}'
    # art_fid_value = (cnt_value + 1) * (fid_value + 1)
    return art_fid_value.item(), fid_value.item(), cnt_value.item(), gray_cnt_value.item(), 

def compute_cfsd(path_to_stylized, path_to_content, batch_size, device, num_workers=1):
    """Computes CFSD for the given paths.

    Args:
        path_to_stylized (str): Path to the stylized images.
        path_to_content (str): Path to the content images.
        batch_size (int): Batch size for computing activations.
        device (str): Device for commputing activations.
        num_workers (int): Number of threads for data loading.

    Returns:
        (float) CFSD value.
    """
    print('Compute CFSD value...')

    simi_val = compute_patch_simi(path_to_stylized, path_to_content, 1, device, num_workers)
    simi_dist = f'{simi_val.item():.4f}'
    return simi_dist

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for computing activations.')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of threads used for data loading.')
    parser.add_argument('--content_metric', type=str, default='lpips', choices=['lpips', 'vgg', 'alexnet', 'ssim', 'ms-ssim'], help='Content distance.')
    parser.add_argument('--mode', type=str, default='art_fid_inf', choices=['art_fid', 'art_fid_inf'], help='Evaluate ArtFID or ArtFID_infinity.')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use.')
    parser.add_argument('--sty', type=str, required=True, help='Path to style images.')
    parser.add_argument('--cnt', type=str, required=True, help='Path to content images.')
    parser.add_argument('--tar', type=str, required=True, help='Path to stylized images.')
    args = parser.parse_args()

    artfid, fid, lpips, lpips_gray = compute_art_fid(args.tar,
                                                    args.sty,
                                                    args.cnt,
                                                    args.batch_size,
                                                    args.device,
                                                    args.mode,
                                                    args.content_metric,
                                                    args.num_workers)

    cfsd = compute_cfsd(args.tar,
                        args.cnt,
                        args.batch_size,
                        args.device,
                        args.num_workers)

    print('ArtFID:', artfid, 'FID:', fid, 'LPIPS:', lpips, 'LPIPS_gray:', lpips_gray)
    print('CFSD:', cfsd)

if __name__ == '__main__':
    main()
