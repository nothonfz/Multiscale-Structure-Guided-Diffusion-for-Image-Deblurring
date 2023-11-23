from model.DVSR import DVSr
from GOPRO_dataset import GroPro_dataset, get_file_path
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torchvision.transforms import v2
from REDS_dataset import REDS_dataset
from PIL import Image


def transform_img(img_list, crop_size, mode):
    to_tensor = v2.ToTensor()
    imgs = [to_tensor(img) for img in img_list]
    if mode == 'train':
        stack_img = torch.stack(imgs, 0)
        transforms = v2.Compose([v2.RandomCrop(crop_size), v2.RandomHorizontalFlip(), v2.RandomVerticalFlip()])
        img_t = transforms(stack_img)
        img_t = torch.unbind(img_t, 0)
    else:
        stack_img = torch.stack(imgs, 0)
        transforms = v2.CenterCrop(crop_size)
        img_t = transforms(stack_img)
        img_t = torch.unbind(img_t, 0)
    return img_t


if __name__ == '__main__':
    center_crop_size = 512
    schedule = {"schedule": "linear",
                "n_timestep": 2000,
                "linear_start": 1e-6,
                "linear_end": 1e-2}
    # s_img_path_test, b_img_path_test = get_file_path('images/test')
    # test_dataset = GroPro_dataset(s_img_path_test, b_img_path_test, crop_size=center_crop_size, mode='test')
    # test_dataloader = DataLoader(test_dataset, 1, pin_memory=True)
    # test_dataset = REDS_dataset('REDS/REDS_test', center_crop_size, mode='test')


    model = DVSr(img_size=center_crop_size, device='cuda', loss_type='l1', use_noisy=False, use_lr=False, lr_rate=8)
    checkpoint = torch.load('6400epoch_not_use_noisy.pth', map_location='cuda:2')
    model.load_state_dict(checkpoint['model'], strict=False)
    model.diffusion.set_new_noise_schedule(schedule, device='cuda')
    model = model.cuda()
    writer = SummaryWriter(log_dir='logs_inference')
    model.eval()
    # image_index = [781, 818, 918]
    # image_index = [781, 256, 512, 128]
    image_index = [527]
    GOPRO_img_path = ['images/test/GOPR0410_11_00/blur/000227.png']
    RealBlur_img_path = ['RealBlur/RealBlur-J_ECC_IMCORR_centroid_itensity_ref/scene208/blur/blur_5.png',
                         'RealBlur/RealBlur-J_ECC_IMCORR_centroid_itensity_ref/scene129/blur/blur_15.png',
                         'RealBlur/RealBlur-J_ECC_IMCORR_centroid_itensity_ref/scene112/blur/blur_14.png',
                         'RealBlur/RealBlur-J_ECC_IMCORR_centroid_itensity_ref/scene107/blur/blur_6.png',
                         'RealBlur/RealBlur-J_ECC_IMCORR_centroid_itensity_ref/scene103/blur/blur_6.png',
                         'RealBlur/RealBlur-J_ECC_IMCORR_centroid_itensity_ref/scene024/blur/blur_18.png',
                         ]
    for i in range(len(RealBlur_img_path)):
        blur_img = Image.open(RealBlur_img_path[i])
        img_t = transform_img([blur_img], crop_size=center_crop_size, mode='test')
        # b = img[0]
        # b = b.cuda()
        b = img_t[0].cuda()
        # s = img['sharp'].cuda()
        writer.add_image('blur', b, i)
        # writer.add_image('sharp', s, i)
        # samples = model.diffusion.p_sample_loop(b.reshape(1, 3, center_crop_size, center_crop_size), continous=False)
        # samples = model.diffusion.p_sample_loop(togray(b).unsqueeze(0), continous=False)
        # writer.add_image('3_sample_gray', samples, i)
        samples, start = model.diffusion.p_sample_loop(b.unsqueeze(0), continous=True)
        for j in range(0, start.shape[0]):
            writer.add_image(f'{i}_predict_start', start[j], j)
        writer.add_image('3_sample', samples[-1], i)
        # for size in multi_size:
        #     b_crop = []
        #     for h in range(center_crop_size//size):
        #         for w in range(center_crop_size//size):
        #             b_crop.append(multi_sample[:, h*size:h*size+size, w*size:w*size+size])
        #             # s_crop = s[:, :, h*size:h*size+size, w*size:w*size+size]
        #     stack_img = torch.stack(b_crop, 0)
        #     sample = model.diffusion.p_sample_loop(stack_img, continous=False, stack_input=True)
        #     # multi_sample[:, :, h * size:h * size + size, w * size:w * size + size] = sample
        #     unbind_img = torch.unbind(sample, 0)
        #     index = 0
        #     for h in range(center_crop_size//size):
        #         for w in range(center_crop_size//size):
        #             multi_sample[:, h * size:h * size + size, w * size:w * size + size] = unbind_img[index]
        #             index += 1
        # writer.add_image('2_multi_sample', multi_sample, i)
        # for i in range(0, samples.shape[0]):
        #     writer.add_image('2_samples', samples[i], i)

