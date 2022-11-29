import torch
import numpy as np
import argparse, os, glob, ast
from imageio import imread, imwrite
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

from msanet import MSANet
from utils import create_dir, logger

parser = argparse.ArgumentParser(description='MSANET')
parser.add_argument('--real', action='store_true', help='real world or synthetic noisy image')
parser.add_argument('--gray', action='store_true', help='if gray image, set True')
parser.add_argument('--save_result', action='store_true', help='if save result, set True')
parser.add_argument('--sigma', type=int, default=30, help='Gaussian noise std. for synthetic images')
parser.add_argument('--ckpt_pth', type=str, default='../ckpt/real_J.pth', help='model path')
parser.add_argument('--data_root', type=str, default='dataset/test_real/', help='clear image path')
parser.add_argument('--datasets', type=ast.literal_eval, default=['Nam_PNG'], help='testing datasets')
parser.add_argument('--result_dir', type=str, default='results/', help='results dir')
parser.add_argument('--log_dir', type=str, default='results.txt', help='log file')
args = parser.parse_args()


logging = logger(args.log_dir)

def evaluate_model():
    dataset_folder_list = []
    result_folder_list = []

    for item in args.datasets:
        dataset_folder_list.append(args.data_root + item)
        result_folder_list.append(args.result_dir + item)

    psnr = np.zeros(len(dataset_folder_list))
    ssim = np.zeros(len(dataset_folder_list))

    inp_channel, out_channel = [1, 1] \
        if args.gray else [3, 3]
    model = MSANet(inp_channel, out_channel)
    model.load_state_dict(torch.load(args.ckpt_pth))
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    for i in range(len(dataset_folder_list)):
        create_dir(result_folder_list[i])
        gt_files = sorted(glob.glob(dataset_folder_list[i]+'/clean/*.png'))
        in_files = sorted(glob.glob(dataset_folder_list[i]+'/noise/*.png')) if args.real \
        else sorted(glob.glob(dataset_folder_list[i]+'/sig'+str(args.sigma)+'/*.png'))

        for ind in range(len(in_files)):
            clean = imread(gt_files[ind]).astype(np.float32)/255.
            clean = clean[0:(clean.shape[0]//8)*8, 0:(clean.shape[1]//8)*8]
            noisy = imread(in_files[ind]).astype(np.float32)/255.
            noisy = noisy[0:(noisy.shape[0]//8)*8, 0:(noisy.shape[1]//8)*8]
            
            img_test = transforms.functional.to_tensor(noisy)
            img_test = img_test.unsqueeze_(0).float()
            if torch.cuda.is_available():
                img_test = img_test.cuda()
            
            with torch.no_grad():
                out_image = model(img_test)

            rgb = out_image.cpu().detach().numpy()
            rgb = rgb.transpose((0,2,3,1))
            if noisy.ndim == 3:
                rgb = np.clip(rgb[0], 0, 1)
            elif noisy.ndim == 2:
                rgb = np.clip(rgb[0, :, :, 0], 0, 1)
            
            if args.save_result:
                img_name = os.path.split(in_files[ind])[-1].split('.')[0]
                imwrite(result_folder_list[i] + '/' + img_name + ".png", np.uint8(rgb*255))

            psnr[i] += compare_psnr(clean, rgb, data_range=1.0)
            ssim[i] += compare_ssim(clean, rgb, data_range=1.0, multichannel=not args.gray)

        psnr[i] = psnr[i] / len(in_files)
        ssim[i] = ssim[i] / len(in_files)
    
    for i in range(len(dataset_folder_list)):
        logging.info('Folder: {}, PSNR: {:.3f}, SSIM:{:.5f}'.format(
            dataset_folder_list[i], psnr[i], ssim[i]
        ))

if __name__ == "__main__":
    evaluate_model()