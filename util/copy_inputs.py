import os, shutil, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cnt', default = './data/cnt')
parser.add_argument('--sty', default = './data/sty')
opt = parser.parse_args()

# inputs of directory
a_folder_path = opt.cnt
b_folder_path = opt.sty

# destination directory of copied inputs
result_folder_path = a_folder_path + '_eval'    # "./data/cnt_eval"
result_folder_path_ = b_folder_path + '_eval'   # "./data/sty_eval"

# get images in the directories
a_images = [f for f in os.listdir(a_folder_path) if f.endswith('.png')]
b_images = [f for f in os.listdir(b_folder_path) if f.endswith('.png')]

# if no dst, make directory
if not os.path.exists(result_folder_path):
    os.makedirs(result_folder_path)
if not os.path.exists(result_folder_path_):
    os.makedirs(result_folder_path_)

# copy content and style images by all combination of pairs
for a_image in a_images:
    for b_image in b_images:
        a_image_path = os.path.join(a_folder_path, a_image)
        b_image_path = os.path.join(b_folder_path, b_image)
        # save the content image
        result_filename = f"{os.path.splitext(a_image)[0]}_stylized_{os.path.splitext(b_image)[0]}.png"
        result_path = os.path.join(result_folder_path, result_filename)
        shutil.copy(a_image_path, result_path)

for a_image in a_images:
    for b_image in b_images:
        a_image_path = os.path.join(a_folder_path, a_image)
        b_image_path = os.path.join(b_folder_path, b_image)
        # save the content image
        result_filename = f"{os.path.splitext(a_image)[0]}_stylized_{os.path.splitext(b_image)[0]}.png"
        result_path = os.path.join(result_folder_path_, result_filename)
        shutil.copy(b_image_path, result_path)