import os
import imageio
import time
import torch.nn as nn

from tqdm import tqdm, trange

from argparser import config_parser
from load_blender import load_blender_data
from NeRF import NeRF
from nerf_utils import *
from nerf_ray_generate import generate_ray_batch_train, generate_ray_batch_test
from nerf_ray_generate import generate_coarse_samples, generate_fine_samples

np.random.seed(0)
DEBUG = False


def run_network(pts, view_dir, model, chunk=1024 * 64):
    xx = pts
    pts_flatten = pts.view(pts.shape[0] * pts.shape[1], pts.shape[2])  # (N, 64, 3) -> (N * 64, 3)
    view_dir = view_dir.view(view_dir.shape[0], 1, view_dir.shape[1])  # (N, 3) -> (N, 1, 3)
    view_dir = view_dir.repeat(1, xx.shape[1], 1)  # (N, 1, 3) -> (N, 64, 3)
    view_dir = view_dir.view(view_dir.shape[0] * view_dir.shape[1], view_dir.shape[2])  # (N, 64, 3) -> (N * 64, 3)

    outputs_flat = torch.cat([model(pts_flatten[i:i + chunk], view_dir[i:i + chunk])
                              for i in range(0, pts_flatten.shape[0], chunk)], 0)
    outputs = torch.reshape(outputs_flat, list(pts.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024 * 32, **kwargs):
    """
    Render rays in smaller mini batches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i + chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(ray_batch, ray_shape=None, chunk=1024 * 32, **kwargs):
    """
    Render rays
    Args:
      ray_batch: array of shape [B, 11]. The flattened composite ray tensor generated with utility functions
      ray_shape: array of shape [H, W, 3]. This parameter is only passed in during test rendering. It tells
        the render function to reshape the flattened output tensor to the image shape.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if ray_shape is None:
        rays_d = ray_batch[..., 3:6]
        sh = rays_d.shape  # [..., 3]
    else:
        sh = ray_shape

    # Render and reshape
    all_ret = batchify_rays(ray_batch, chunk, **kwargs)

    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, k, chunk, render_kwargs, gt_imgs=None, save_dir=None, render_factor=0):
    h, w, focal = hwf

    if render_factor != 0:
        # Render downsampled for speed
        h = h // render_factor
        w = w // render_factor
        focal = focal / render_factor
        hwf = (h, w, focal)

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        ray_batch = generate_ray_batch_test(hwf, k, c2w,
                                            near=render_kwargs["near"],
                                            far=render_kwargs["far"],
                                            ndc=render_kwargs["ndc"])

        ray_d_flatten = ray_batch[..., 3:6]
        ray_d = ray_d_flatten.reshape(h, w, -1)     # should be [H, W, 3]
        rgb, disp, acc, _ = render(ray_batch, ray_shape=ray_d.shape, chunk=chunk, **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i == 0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if save_dir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(save_dir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(args):
    """
    Instantiate NeRF's MLP model.
    """
    model = NeRF(device=device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(device=device)
        grad_vars += list(model_fine.parameters())

    def network_query_fn(inputs, view_dir, network_fn):
        return run_network(inputs, view_dir, network_fn, chunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f)
                 for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_coarse': model,
        'network_fine': model_fine,
        'network_query_fn': network_query_fn,
        'n_coarse_sample': args.N_samples,
        'n_fine_sample': args.N_importance,
        'perturb': args.perturb,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0., white_bkgd=False):
    """
    Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
        raw_noise_std: float. The noise to add to the raw rgb output.
        white_bkgd: boolean. If True, the image is rendered with white background.
            The parameter is for blender dataset only
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """

    def raw2alpha(raw_, dists_, act_fn=nn.functional.relu):
        return 1. - torch.exp(-act_fn(raw_) * dists_)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std

    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch,
                network_coarse,
                network_fine,
                network_query_fn,
                n_coarse_sample,
                n_fine_sample,
                lindisp=False,
                perturb=0.,
                white_bkgd=False,
                raw_noise_std=0.,
                **kwargs):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_coarse: function. Model for predicting RGB and density at each point
        in space.
      network_fine: "fine" network with same spec as network_fn.
      network_query_fn: function used for passing queries to network_fn.
      n_coarse_sample: int. Number of different times to sample along each ray.
      n_fine_sample: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    rays_d = ray_batch[:, 3:6]  # [N_rays, 3] each

    # generate coarse samples
    pts_coarse, view_dir_coarse, z_vals_coarse = generate_coarse_samples(ray_batch, n_coarse_sample,
                                                                         inv_depth=lindisp, perturb=perturb)

    # query the coarse network
    raw = network_query_fn(pts_coarse, view_dir_coarse, network_coarse)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals_coarse, rays_d, raw_noise_std, white_bkgd)

    rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

    # generate fine samples
    pts, view_dir, z_vals, z_samples = generate_fine_samples(ray_batch, z_vals_coarse, weights, n_fine_sample,
                                                             perturb=perturb)

    # query the fine network
    run_fn = network_coarse if network_fine is None else network_fine
    raw = network_query_fn(pts, view_dir, run_fn)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd)

    ret = {
        'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map,
        'rgb0': rgb_map_0, 'disp0': disp_map_0, 'acc0': acc_map_0,
        'z_std': torch.std(z_samples, dim=-1, unbiased=False)
    }

    if DEBUG:
        for k in ret:
            if torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any():
                print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def train():
    parser = config_parser()
    args = parser.parse_args()

    # Load data
    k = None
    if args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split, bounding_box = load_blender_data(args.datadir, args.half_res,
                                                                                    args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
        else:
            images = images[..., :3]

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    h, w, focal = hwf
    h, w = int(h), int(w)
    hwf = [h, w, focal]

    if k is None:
        k = np.array([
            [focal, 0, 0.5 * w],
            [0, focal, 0.5 * h],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
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

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            test_save_dir = os.path.join(basedir, expname,
                                         'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(test_save_dir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _ = render_path(render_poses, hwf, k, args.chunk, render_kwargs_test, gt_imgs=images,
                                  save_dir=test_save_dir, render_factor=args.render_factor)
            print('Done rendering', test_save_dir)
            imageio.mimwrite(os.path.join(test_save_dir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

    # Prepare ray batch tensor if batching random rays
    N_rand = args.N_rand

    poses = torch.Tensor(poses).to(device)

    n_iters = 30000 + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    start = start + 1
    for i in trange(start, n_iters):
        time0 = time.time()

        # Random from one image
        img_i = np.random.choice(i_train)
        target = images[img_i]
        target = torch.Tensor(target).to(device)
        pose = poses[img_i, :3, :4]
        ray_batch, target_rgb = generate_ray_batch_train(target, pose,
                                                         near, far, hwf, k, N_rand,
                                                         curr_step=i, ndc=render_kwargs_train["ndc"],
                                                         pre_crop_iter=args.precrop_iters,
                                                         pre_crop_frac=args.precrop_frac)

        # Core optimization loop #
        rgb, disp, acc, extras = render(ray_batch, chunk=args.chunk, **render_kwargs_train)

        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_rgb)
        loss = img_loss
        psnr = mse2psnr(img_loss)

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_rgb)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        # update learning rate #
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time() - time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        # end #

        # Rest is logging
        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i % args.i_video == 0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, k, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            movie_base = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(movie_base + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(movie_base + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

        # if args.use_viewdirs:
        #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
        #     with torch.no_grad():
        #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
        #     render_kwargs_test['c2w_staticcam'] = None
        #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        if i % args.i_testset == 0 and i > 0:
            test_save_dir = os.path.join(basedir, expname, 'test_{:06d}'.format(i))
            os.makedirs(test_save_dir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test[:4]]).to(device), hwf, k, args.chunk, render_kwargs_test,
                            gt_imgs=images[i_test], save_dir=test_save_dir, render_factor=args.render_factor)
            print('Saved test set')

        if i % args.i_print == 0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")

        global_step += 1


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
