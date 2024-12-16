# [CVPR 2024 Highlight]  Diffuser implementation of "Style Injection in Diffusion: A Training-free Approach for Adapting Large-scale Diffusion Models for Style Transfer"

 
### [Paper](https://arxiv.org/abs/2312.09008) / [Project Page](https://jiwoogit.github.io/StyleID_site/)

## Notices
This is implementation of StyleID based on [diffusers](https://github.com/huggingface/diffusers) library. 
You may refer to the original implementation for obtaining quantitative metrics reported in the paper.

### Update: Support for Stable Diffusion 2.1 (2024-12-14)

Thanks to [@LiManyuan663](https://github.com/LiManyuan663) the code now supports running with `sd_version=2.1`. You can use the updated script as shown below:

```bash
python run_styleid_diffusers.py --cnt_fn data/cnt.png --sty_fn data/sty.png --sd_version 2.1-base
python run_styleid_diffusers.py --cnt_fn data/cnt.png --sty_fn data/sty.png --sd_version 2.1
```

## Run StyleID
For running StyleID, run:

```
python run_styleid_diffusers.py --cnt_fn <content_img_path> --sty_fn <style_img_path>
```
For running default configuration in sample image files, run:
```
python run_styleid_diffusers.py --cnt_fn data/cnt.png --sty_fn data/sty.png --gamma 0.75 --T 1.5  # default
python run_styleid_diffusers.py --cnt_fn data/cnt.png --sty_fn data/sty.png --gamma 0.3 --T 1.5   # high style fidelity
```

To fine-tune the parameters, you have control over the following aspects in the style transfer:

- **Attention-based style injection** is removed by the `--without_attn_injection` parameter.
- **Query preservation** is controlled by the `--gamma` parameter.
  (A higher value enhances content fidelity but may result a lack of style fidelity).
- **Attention temperature scaling** is controlled through the `--T` parameter.
- **Initial latent AdaIN** is removed by the `--without_init_adain` parameter.



## Citation
If you find our work useful, please consider citing and star:
```BibTeX

@article{chung2023style,
  title={Style Injection in Diffusion: A Training-free Approach for Adapting Large-scale Diffusion Models for Style Transfer},
  author={Chung, Jiwoo and Hyun, Sangeek and Heo, Jae-Pil},
  journal={arXiv preprint arXiv:2312.09008},
  year={2023}
}
```
