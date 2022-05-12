import os
import glob
import argparse
from tqdm import tqdm
import json
import pickle
import random
import oyaml as yaml

# for the model
import torch
import torchvision
from sgan.stylegan2_utils import renormalize, nethook
from stylegan_model import Generator
from PIL import Image
import numpy as np
from scipy.stats import truncnorm

if torch.cuda.is_available():
    print('cuda is available.')
    device = 'cuda'
else:
    print('No cuda available!')
    device = 'cpu'

def truncated_noise_sample_neighbors(batch_size=1, dim_z=512, truncation=1., seed=None, num_neighbors=0, scale=0.25):
    """ Create a truncated noise vector.
        Params:
            batch_size: batch size.
            dim_z: dimension of z
            truncation: truncation value to use
            seed: seed for the random generator
        Output:
            array of shape (batch_size, dim_z)
    """
    list_results = []

    if truncation is not None:
        state = None if seed is None else np.random.RandomState(seed)

        zs = truncation * torch.from_numpy(truncnorm.rvs(-2, 2, size=(batch_size, dim_z), random_state=state).astype(np.float32)).to(device)
        list_results.append(zs) # these are anchors

        for i in range(num_neighbors):
            state_neighbors = None if seed is None else np.random.RandomState(seed+1000+i)
            values_neighbors = truncation * torch.from_numpy(truncnorm.rvs(-2, 2, size=(batch_size, dim_z),
                                                          scale=scale, random_state=state_neighbors).astype(np.float32)).to(device)

            list_results.append(zs + values_neighbors)    
    else:
        np.random.seed(seed)
        zs = torch.tensor(np.random.normal(0.0, 1.0, [batch_size, dim_z]), dtype=torch.float32, requires_grad=False).to(device)
        list_results.append(zs) # these are anchors

        for i in range(num_neighbors):
            np.random.seed(seed+1000+i)
            values_neighbors = torch.tensor(np.random.normal(0.0, scale, [batch_size, dim_z]), dtype=torch.float32,
                                            requires_grad=False).to(device)

            list_results.append(zs + values_neighbors)

    return list_results
    
def sample(opt):
    if 'cifar' in opt.dataset_type:
        image_size = 32
    elif opt.dataset_type == 'tinyimagenet':
        image_size = 64

    # load the model
    g_ckpt = torch.load(opt.checkpoint_path, map_location=device)
    latent_dim = g_ckpt['args'].latent
    gan_model = Generator(image_size, latent_dim, 8).to(device)
    gan_model.load_state_dict(g_ckpt["g_ema"], strict=False)
    gan_model.eval()
    nethook.set_requires_grad(False, gan_model)
    print('GAN network loaded')

    output_path = opt.output_path
    partition = opt.partition
    # start_seed, nimg = constants.get_seed_nimg(partition)
    start_seed = opt.start_seed
    nimg = opt.num_imgs

    batch_size = opt.batch_size
    if opt.dataset_type == 'cifar100':
        ds = torchvision.datasets.CIFAR100(opt.data_dir, train=True, download=True)
        class_index = ds.class_to_idx
        del ds
    
    class_index_keys = list(class_index.keys())
        
    random.shuffle(class_index_keys)
    for key in tqdm(class_index_keys): 
        class_dir_name = os.path.join(output_path, partition, str(class_index[key]))
        if os.path.isdir(class_dir_name):
            continue
        os.makedirs(class_dir_name, exist_ok=True)
        idx = int(class_index[key])
        z_dict = dict()
                
        print('Generating images for class {}, with number of images to be {}'.format(idx, nimg))
        seed = start_seed + idx
        noise_vector_neighbors = truncated_noise_sample_neighbors(batch_size=nimg,
                                                                  dim_z=latent_dim,
                                                                  seed=seed, 
                                                                  truncation = opt.truncation,
                                                                  num_neighbors=opt.num_neighbors,
                                                                  scale=opt.std)
        for ii in range(len(noise_vector_neighbors)):
            noise_vector = noise_vector_neighbors[ii]
            for batch_start in range(0, nimg, batch_size):
                s = slice(batch_start, min(nimg, batch_start + batch_size))
                with torch.no_grad():
                    ims, _ = gan_model([noise_vector[s]],
                                    input_is_latent=False,
                                    randomize_noise=True)
    
                for i, im in enumerate(ims):
                    if ii == 0: #anchors
                        im_name = 'seed%04d_sample%05d_anchor.%s' % (seed, batch_start+i, opt.imformat)
                    else:
                        im_name = 'seed%04d_sample%05d_1.0_%d.%s' % (seed, batch_start+i, ii, opt.imformat)

                    im = renormalize.as_image(im)
                    im = Image.fromarray(im)
                    im.save(os.path.join(class_dir_name, im_name))
                    z_dict[im_name] = [noise_vector[batch_start+i].cpu().numpy(), idx]
        with open(os.path.join(class_dir_name, 'z_dataset.pkl'), 'wb') as fid:
            pickle.dump(z_dict,fid)                                                        
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Sample from stylegan2 (new)")
    parser.add_argument('--out_dir', default='/disk_d/han/data/', type=str)
    parser.add_argument('--data_dir', default='/disk_d/han/data/', type=str)
    parser.add_argument('--checkpoint_path', default='/', type=str)
    parser.add_argument('--partition', default='train', type=str)
    parser.add_argument('--truncation', default=0.9, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--imformat', default='png', type=str)
    parser.add_argument('--num_imgs', default=1300, type=int, help='num imgs per class')
    parser.add_argument('--start_seed', default=0, type=int)
    parser.add_argument('--dataset_type', default='cifar100', type=str, choices=['cifar10', 'cifar100', 'tinyimagenet'],help='choices: 100, 893, or 1000, equivalent to number of images in image100, cars, or imagenet1000')
    parser.add_argument('--num_neighbors', default=1, type=int, help='num samples per anchor')
    parser.add_argument('--std', default=0.20, type=float, help='std for gaussian in z space')

    opt = parser.parse_args()
    if opt.num_neighbors == 0:
        output_path = (os.path.join(opt.out_dir, 'stylegan_{}_tr{}'.format(opt.dataset_type, opt.truncation)))
        opt.std = 1.0 # we only sample from normal
    else:
        output_path = (os.path.join(opt.out_dir, 
                       'stylegan_{}_tr{}_gauss1_std{}_NS{}_NN{}'.format(opt.dataset_type, opt.truncation, 
                                                                       opt.std, opt.num_imgs, opt.num_neighbors)))
    opt.output_path = output_path
    print(opt)
    if not os.path.isdir(opt.output_path):
        os.makedirs(opt.output_path, exist_ok=True)
    with open(os.path.join(opt.output_path, 'opt_summary.yml'), 'w') as fid:
        yaml.dump(vars(opt), fid, default_flow_style=False)
    sample(opt)