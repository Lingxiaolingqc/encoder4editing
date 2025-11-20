
import argparse
print("0")
import torch
print("1")
import numpy as np
print("2")
import sys
print("3")
import os
print("4")
import dlib
print("5")
sys.path.append(".")
sys.path.append("..")
print("6")
from configs import data_configs, paths_config
print("7")
from datasets.inference_dataset import InferenceDataset
print("8")
from torch.utils.data import DataLoader
print("9")
from utils.model_utils import setup_model
print("10")
from utils.common import tensor2im
print("11")
from utils.alignment import align_face
print("12")
from PIL import Image
print("13")

def main(args):
    net, opts = setup_model(args.ckpt, device)
    is_cars = 'cars_' in opts.dataset_type
    generator = net.decoder
    generator.eval()
    args, data_loader = setup_data_loader(args, opts)
    # Check if latents exist

    # # change
    # if os.path.exists(latents_file_path):
    #     latent_codes = torch.load(latents_file_path).to(device)
    # else:
    #     latent_codes = get_all_latents(net, data_loader, args.n_sample, is_cars=is_cars)
    #     torch.save(latent_codes, latents_file_path)

    latent_codes,paths = get_all_latents(net, data_loader, args , args.n_sample, is_cars=is_cars)

    if not args.latents_only:
        generate_inversions(args, generator, latent_codes,paths, is_cars=is_cars)


def setup_data_loader(args, opts):
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    images_path = args.images_dir if args.images_dir is not None else dataset_args['test_source_root']
    print(f"images path: {images_path}")
    align_function = None
    if args.align:
        align_function = run_alignment
    test_dataset = InferenceDataset(root=images_path,
                                    transform=transforms_dict['transform_test'],
                                    preprocess=align_function,
                                    opts=opts)

    data_loader = DataLoader(test_dataset,
                             batch_size=args.batch,
                             shuffle=False,
                             num_workers=2,
                             drop_last=True)

    print(f'dataset length: {len(test_dataset)}')

    if args.n_sample is None:
        args.n_sample = len(test_dataset)
    return args, data_loader


def get_latents(net, x, is_cars=False):
    codes = net.encoder(x)
    if net.opts.start_from_latent_avg:
        if codes.ndim == 2:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
        else:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)
    if codes.shape[1] == 18 and is_cars:
        codes = codes[:, :16, :]
    return codes


def get_all_latents(net, data_loader, args, n_images=None, is_cars=False):
    all_latents = []
    all_paths =[]
    i = 0
    j=1
    with torch.no_grad():
        # for batch,path in data_loader:
        #     if n_images is not None and i > n_images:
        #         break
        #     #change
        #     x = batch
        #     inputs = x.to(device).float()
        #     latents = get_latents(net, inputs, is_cars)
        #     all_latents.append(latents)
        #     i += len(latents)
        #     latents_file_path = os.path.join(args.save_dir, f'{j}.pt')
        #     j+=1
        #     torch.save(latents, latents_file_path)
        for imgs, paths in data_loader:
            if n_images is not None and i >= n_images:
                break
            
            inputs = imgs.to(device).float()
            latents = get_latents(net, inputs, is_cars)

            # 对 batch 中的每一张图片单独保存 latent
            for latent, path in zip(latents, paths):
                filename = os.path.basename(path)
                name, _ = os.path.splitext(filename)
                save_path = os.path.join(args.save_dir, f"{name}.pt")
                
                torch.save(latent.unsqueeze(0), save_path)
                all_latents.append(latent.unsqueeze(0))
                all_paths.append(path)
            i += len(paths)

    return torch.cat(all_latents),all_paths


def save_image(img, save_dir, idx):
    result = tensor2im(img)
    im_save_path = os.path.join(save_dir, f"{idx}.jpg")
    # im_save_path = os.path.join(save_dir, f"{idx:05d}.jpg")
    Image.fromarray(np.array(result)).save(im_save_path)


@torch.no_grad()
def generate_inversions(args, g, latent_codes,paths, is_cars):
    print('Saving inversion images')
    inversions_directory_path = os.path.join(args.save_dir, 'inversions')
    os.makedirs(inversions_directory_path, exist_ok=True)
    for latent, path in zip(latent_codes, paths):

        # 取原图文件名（不带后缀）
        filename = os.path.basename(path)
        name, _ = os.path.splitext(filename)

        # 生成 inversion
        imgs, _ = g([latent.unsqueeze(0)], input_is_latent=True, randomize_noise=False, return_latents=True)
        if is_cars:
            imgs = imgs[:, :, 64:448, :]

        # 保存为同名文件
        save_path = os.path.join(inversions_directory_path, f"{name}.jpg")
        save_image(imgs[0], inversions_directory_path, name)
    # for i in range(min(args.n_sample, len(latent_codes))):
    #     imgs, _ = g([latent_codes[i].unsqueeze(0)], input_is_latent=True, randomize_noise=False, return_latents=True)
    #     if is_cars:
    #         imgs = imgs[:, :, 64:448, :]
    #     save_image(imgs[0], inversions_directory_path, i + 1)


def run_alignment(image_path):
    predictor = dlib.shape_predictor(paths_config.model_paths['shape_predictor'])
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image


if __name__ == "__main__":
    device = "cuda"
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--images_dir", type=str, default=None,
                        help="The directory of the images to be inverted")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="The directory to save the latent codes and inversion images. (default: images_dir")
    parser.add_argument("--batch", type=int, default=1, help="batch size for the generator")
    parser.add_argument("--n_sample", type=int, default=None, help="number of the samples to infer.")
    parser.add_argument("--latents_only", action="store_true", help="infer only the latent codes of the directory")
    parser.add_argument("--align", action="store_true", help="align face images before inference")
    parser.add_argument("ckpt", metavar="CHECKPOINT", help="path to generator checkpoint")

    args = parser.parse_args()
    main(args)
