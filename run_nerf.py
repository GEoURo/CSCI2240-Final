import os, sys
from datetime import datetime
import numpy as np
import imageio
import json
import pdb
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm, trange
import pickle

from argparser import config_parser
from load_blender import load_blender_data
import parser

np.random.seed(0)


def train():
    parser = config_parser()
    args = parser.parse_args()

    # load data
    if args.dataset_type == "blender":
        images, poses, render_poses, hwf, i_split, bounding_box = load_blender_data(args.datadir, args.half_res,
                                                                                    args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)

        args.bounding_box = bounding_box

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
        else:
            images = images[..., :3]

    else:
        print("unsupported dataset type")
        return

    i_train, i_val, i_test = i_split

    # Cast intrinsics to right types
    h, w, focal = hwf
    h, w = int(h), int(w)
    hwf = [h, w, focal]

    k = np.array([
            [focal, 0, 0.5*w],
            [0, focal, 0.5*h],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    if args.i_embed == 1:
        args.expname += "_hashXYZ"
    elif args.i_embed == 0:
        args.expname += "_posXYZ"
    if args.i_embed_views == 2:
        args.expname += "_sphereVIEW"
    elif args.i_embed_views == 0:
        args.expname += "_posVIEW"
    args.expname += "_fine" + str(args.finest_res) + "_log2T" + str(args.log2_hashmap_size)
    args.expname += "_lr" + str(args.lrate) + "_decay" + str(args.lrate_decay)
    args.expname += "_Adam"
    if args.sparse_loss_weight > 0:
        args.expname += "_sparse" + str(args.sparse_loss_weight)
    args.expname += "_TV" + str(args.tv_loss_weight)
    args.expname += datetime.now().strftime('_%H_%M_%d_%m_%Y')
    expname = args.expname

    # create argument file
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))

    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

if __name__ == '__main__':
    train()