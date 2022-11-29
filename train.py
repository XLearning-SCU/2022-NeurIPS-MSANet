import numpy as np
import argparse, os, torch, ast
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

from msanet import MSANet
from utils import create_dir, logger, RandomRot
from dataloader import Dataset_from_syn, Dataset_for_eval, Dataset_h5_real

parser = argparse.ArgumentParser(description='MSANET')
parser.add_argument('--real', type=bool, default=True, help='if real noise images, set True')
parser.add_argument('--gray', type=bool, default=False, help='if grayscale noise images, set True')

parser.add_argument('--src_data', type=str, 
    default='dataset/train_real/train.h5', # real noise images, 
    # default='dataset/train_syn/DIV2K_Sub_Images/*.png', # synthetic noise training images
help='training dataset')
parser.add_argument('--val_data', type=ast.literal_eval, default=[
    'dataset/test_real/SIDD_VAL_100/clean/*.png',
    'dataset/test_real/SIDD_VAL_100/noise/*.png'
], help='validating dataset [clean images, noise images]')
parser.add_argument('--sigma', type=int, default=30, help='synthetic Gaussian noise level')

parser.add_argument('--ckpt_dir', type=str, default='ckpt/real/', help='checkpoints dir')
parser.add_argument('--log_file', type=str, default='log/real.txt', help='log file')

parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--patch_size', type=int, default=128, help='patch size')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--loss', type=str, default='L1', help='training loss')
parser.add_argument('--n_epoch', type=int, default=100, help='the number of epochs')
parser.add_argument('--val_epoch', type=int, default=1, help='do validation per every N epochs')
parser.add_argument('--save_epoch', type=int, default=1, help='save model per every N epochs')
args = parser.parse_args()


def train():
    create_dir(args.ckpt_dir)
    logging = logger(args.log_file)
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    
    if args.real:
        dataset = Dataset_h5_real(args.src_data, args.patch_size)
    else:
        dataset = Dataset_from_syn(args.src_data, args.sigma, args.gray, transforms.Compose([
            transforms.RandomCrop((args.patch_size, args.patch_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Lambda(lambda img: RandomRot(img)),
            transforms.ToTensor()
        ]))
    dataloader = DataLoader(dataset, args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    
    dataset_val = Dataset_for_eval(args.val_data, args.gray)
    dataloader_val = DataLoader(dataset_val, 1, shuffle=False, num_workers=0, drop_last=False)
    
    inp_channel, out_channel = [1, 1] if args.gray else [3, 3]
    model = MSANet(inp_channel, out_channel)

    logging.info('Available GPUs: {}'.format(torch.cuda.device_count()))
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda() \
            if torch.cuda.device_count() > 1 else model.cuda()
    
    if args.loss == 'L2':
        criterion = torch.nn.MSELoss()
    elif args.loss == 'L1':
        criterion = torch.nn.L1Loss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, args.n_epoch)

    for epoch in range(args.n_epoch):
        logging.info('Epoch {}, lr {}'.format(epoch+1, optimizer.param_groups[0]['lr']))
        
        model.train()
        loss_sum = 0
        for i, (inputs, labels) in enumerate(dataloader):
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda().detach(), labels.cuda().detach()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_sum += loss.item()
            if (i != 0) and (i % 100 == 0):
                loss_avg = loss_sum / 100
                logging.info("Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.8f}".format(
                    epoch + 1, args.n_epoch, i + 1, len(dataloader), loss_avg
                ))
                loss_sum = 0.0
        
        scheduler.step()

        if epoch % args.save_epoch == 0:
            state_dict = model.module.state_dict() \
                if torch.cuda.device_count() > 1 else model.state_dict()
            torch.save(state_dict, os.path.join(args.ckpt_dir, 'model_%04d_dict.pth' % (epoch+1)))
        
        if epoch % args.val_epoch == 0:
            model.eval()
            psnr, ssim, loss_val = 0, 0, 0
            for inputs, labels in dataloader_val:
                if torch.cuda.is_available():
                    inputs, labels = inputs.cuda(), labels.cuda()
                
                with torch.no_grad():
                    test_out = model.module(inputs) \
                        if torch.cuda.device_count() > 1 else model(inputs)
                
                loss_val += criterion(test_out, labels).item()
                rgb_out = test_out.cpu().numpy().transpose((0,2,3,1))
                clean = labels.cpu().numpy().transpose((0,2,3,1))
                if args.gray:
                    denoised = np.clip(rgb_out[0,:,:,0], 0, 1)
                    clean = clean[0,:,:,0]
                else:
                    denoised = np.clip(rgb_out[0], 0, 1)
                    clean = clean[0]
                
                psnr += compare_psnr(clean, denoised, data_range=1.0)
                ssim += compare_ssim(clean, denoised, data_range=1.0, multichannel=not args.gray)
            
            loss_val = loss_val / len(dataloader_val)
            psnr = psnr / len(dataloader_val)
            ssim = ssim / len(dataloader_val)
            logging.info('Loss: {:.8f}, PSNR: {:.3f}, SSIM:{:.5f}'.format(loss_val, psnr, ssim))

if __name__ == '__main__':
    train()