PyTorch implementation for [Multi-Scale Adaptive Network for Single Image Denoising](http://pengxi.me/wp-content/uploads/2022/11/MSANet.pdf) (NeurIPS 2022).

## Requirements
- Python 3.6.13
- Pytorch 1.9.0
- mmcv 1.3.14
- h5py, pillow, numpy, scikit-image, *etc.*

## Testing
First, for testing on real noise images, please organize each test dataset as follows
```
|--test_dataset
|   |--clean
|   |   |--*.png
|   |--noise
|   |   |--*.png
```
and run test.py through
```
python test.py \
        --real \                                # flag of real noise images
        --save_result \                         # flag of saving the denoised results
        --ckpt_pth ckpt/real_with_jpeg.pth \    # path to models
        --data_root dataset/test/ \             # path to datasets
        --datasets "['Nam_PNG']"                # list of datasets
```
For testing on synthetic noise images, please organize each test dataset as follows
```
|--test_dataset
|   |--clean
|   |   |--*.png
|   |--sig30
|   |   |--*.png
|   |--sig50
|   |   |--*.png
|   |--sig70
|   |   |--*.png
```
and run test.py for color noise image denoising through
```
python test.py \
        --save_result \                         # flag of saving the denoised results
        --sigma 30 \                            # noise level
        --ckpt_pth ckpt/color_sig30.pth \       # path to models
        --data_root dataset/test/ \             # path to datasets
        --datasets "['CMcMaster']"              # list of datasets
```
as well as grayscale noise image denoising through
```
python test.py \
        --gray  \                               # flag of grayscale noise images
        --save_result \                         # flag of saving the denoised results
        --sigma 30 \                            # noise level
        --ckpt_pth ckpt/gray_sig30.pth \        # path to models
        --data_root dataset/test/ \             # path to datasets
        --datasets "['GMcMaster']"              # list of datasets
```

## Training
To train your own models, please modify the arguments in the train.py and run it through
```
python train.py
```

## Citation
If this work is helpful, please cite it, thanks! >_<
```
@inproceedings{msanet,
  title={Multi-Scale Adaptive Network for Single Image Denoising},
  author={Yuanbiao Gou and Peng Hu and Jiancheng Lv and Joey Tianyi Zhou and Xi Peng},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
```

## Acknowledgement
This work uses some packages from [mmcv](https://github.com/open-mmlab/mmcv) in the implementation, thanks for their excellent work!