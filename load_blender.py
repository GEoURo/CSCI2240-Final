import os
import torch
import numpy as np
import imageio
import json
import cv2

from utils import get_bbox3d_for_blenderobj

trans_t = lambda t: torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()

rot_phi = lambda phi: torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).float()

rot_theta = lambda th: torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w


def load_blender_data(basedir, half_res=False, testskip=1, use_aux_params=False):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    all_aux_scene_params = []
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        aux_scene_params = []
        if s == 'train' or testskip == 0:
            skip = 1
        else:
            skip = testskip

        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
            # Light intensity:
            # aux_scene_params.append(frame['light_intensity'] / 10.0)

            # light position:
            # light_pos = np.array(frame['light_pos'])
            # light_pos[0] = (light_pos[0] + 0.25) / (0.5)
            # light_pos[1] = (light_pos[1] + 0.25) / (0.5)
            # aux_scene_params.append(light_pos)

            # diffuse color:
            # aux_scene_params.append(frame['diffuse'])

            # moving object:
            obj_pos = np.array(frame['obj_pos'])
            obj_pos[0] = (obj_pos[0] + 0.1) / (0.2)
            obj_pos[1] = (obj_pos[1] + 0.1) / (0.2)
            aux_scene_params.append(obj_pos)

        imgs = (np.array(imgs) / 255.).astype(np.float32)  # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
        all_aux_scene_params.append(aux_scene_params)

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    aux_scene_params = np.concatenate(all_aux_scene_params, 0)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    near = 0.1
    far = 2
    radius = (near + far) / 2

    # orbiting camera video:
    # render_poses = torch.stack([pose_spherical(angle, -30.0, radius) for angle in np.linspace(-180, 180, 80 + 1)[:-1]], 0)

    # static video pose:
    render_poses = torch.stack([pose_spherical(angle, -30.0, radius) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0)
    render_poses = render_poses[0:1, :, :]
    render_poses = render_poses.expand(30, -1, -1)
    
    bounding_box = get_bbox3d_for_blenderobj(metas["train"], H, W, near=near, far=far)

    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    if use_aux_params:
        return imgs, poses, render_poses, [H, W, focal], i_split, bounding_box, near, far, aux_scene_params

    return imgs, poses, render_poses, [H, W, focal], i_split, bounding_box, near, far, None
