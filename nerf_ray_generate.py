from hashlib import new
import os
import torch
import imageio
import torch.nn.functional as F

from NeRF import CreateEmbeddingFunction, NeRF

from nerf_utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw = raw.to(device)
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists).to(device)
    
    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)
    dists = dists.to(device)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]

    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    z_vals = z_vals.to(device)
    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map

def render_rays(ray_batch,
                model_query,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                model_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]


#     raw = run_network(pts)
    raw = model_query(pts, viewdirs)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_model = model_query if model_fine is None else model_fine
        # run_fn = network_fn if network_fine is None else network_fine
#         raw = run_network(pts, fn=run_fn)
        raw = run_model(pts, viewdirs)
        # raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()):
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret

def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret

def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if rays is None:
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]

def generate_ray_batch_test(hwf, k, c2w, near, far, ndc=True):
    """
    Generate a batch of rays for testing

    :param hwf: tuple. A tuple of (height, width, focal)
    :param k: array of shape (3, 3). The intrinsic matrix of the camera
    :param c2w: array of shape (3, 4). The camera's transformation matrix.
    :param near: float. The near plane
    :param far: float. The far plane
    :param ndc: bool. If True, represent ray origin, direction in NDC coordinates.
    :return: Tensor of shape (h * w, 11). [ray_o, ray_d, near, far, view_dir]
    """
    h, w, focal = hwf
    rays_o, rays_d = get_rays(h, w, k, c2w)

    view_dir = rays_d
    # normalize the view directions
    view_dir = view_dir / torch.norm(view_dir, dim=-1, keepdim=True)
    view_dir = torch.reshape(view_dir, [-1, 3]).float()

    if ndc:
        rays_o, rays_d = ndc_rays(h, w, focal, 1., rays_o, rays_d)

    # reshape the to get rays
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
    return torch.cat([rays_o, rays_d, near, far, view_dir], dim=-1)


def generate_ray_batch_train(images, poses,
                             near, far, hwf, k,
                             n_rand,
                             curr_step,
                             ndc=True,
                             pre_crop_iter=0,
                             pre_crop_frac=0.5):
    """
    Generate a batch of rays for training.

    The function will randomly choose an image and generate rays

    :param images: A list of images to choose from
    :param poses: A list of camera poses corresponding to each image
    :param near: float. The near plane for rendering.
    :param far: float. The far plane for rendering
    :param hwf: tuple. A tuple of (height, width, focal)
    :param k: array of shape (3, 4). The camera's intrinsic matrix
    :param n_rand: int. The number of ray sample
    :param curr_step: int. Current step for training
    :param ndc: bool. If True, represent ray origin, direction in NDC coordinates.
    :param pre_crop_iter: int. Number of steps to train on central crops
    :param pre_crop_frac: float. Fraction of a image taken for central crops
    :return: [rays_o, rays_d, near, far, view_dir], target_rgb
    """
    image_index = np.random.choice(images.shape[0])
    target = images[image_index]
    target = torch.Tensor(target).to(device)
    pose = poses[image_index, :3, :4]

    h, w, focal = hwf

    rays_o, rays_d = get_rays(h, w, k, torch.Tensor(pose))
    if curr_step < pre_crop_iter:
        d_h = int(h // 2 * pre_crop_frac)
        d_w = int(w // 2 * pre_crop_frac)
        # (h, w, 2)
        coords = torch.stack(
            torch.meshgrid(
                torch.linspace(h // 2 - d_h, h // 2 + d_h - 1, 2 * d_h),
                torch.linspace(w // 2 - d_w, w // 2 + d_w - 1, 2 * d_w)
            ), -1)
    else:
        # (h, w, 2)
        coords = torch.stack(
            torch.meshgrid(
                torch.linspace(0, h - 1, h),
                torch.linspace(0, w - 1, w)
            ), -1)

    coords = torch.reshape(coords, [-1, 2])
    selected_indices = np.random.choice(coords.shape[0], size=[n_rand], replace=False)
    selected_coords = coords[selected_indices].long()  # (N_rand, 2)

    rays_o = rays_o[selected_coords[:, 0], selected_coords[:, 1]]  # (N_rand, 3)
    rays_d = rays_d[selected_coords[:, 0], selected_coords[:, 1]]  # (N_rand, 3)
    target_rgb = target[selected_coords[:, 0], selected_coords[:, 1]]  # (N_rand, 3)

    view_dir = rays_d
    # normalize the view directions
    view_dir = view_dir / torch.norm(view_dir, dim=-1, keepdim=True)
    view_dir = torch.reshape(view_dir, [-1, 3]).float()

    if ndc:
        rays_o, rays_d = ndc_rays(h, w, focal, 1., rays_o, rays_d)

    # reshape the to get rays
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
    return torch.cat([rays_o, rays_d, near, far, view_dir], dim=-1), target_rgb


def generate_coarse_samples(ray_batch, n_sample, inv_depth=False, perturb=True):
    """
    Generate point samples for a batch of ray
    :param ray_batch: Tensor. The compact ray batch returned from generate_ray_batch functions.
                      Must have the shape of (N, 3)
    :param n_sample: int. The number of samples taken along the ray
    :param inv_depth: bool. If True, sample linearly in inverse depth rather than in depth.
    :param perturb: bool. If True, each ray is sampled at stratified random points.
    :return: pts, view_dir, z_vals.
             pts.shape      = (n_rays, n_sample, 3)
             view_dir.shape = (n_rays, 3)
             z_vals         = (n_rays, n_samples)
    """
    assert ray_batch.shape[-1] == 11

    n_rays = ray_batch.shape[0]
    # extract origin and directions of the rays
    rays_o, rays_d = ray_batch[..., 0:3], ray_batch[..., 3:6]  # (N_rays, 3) each
    # extract near and far plane for rendering
    near, far = ray_batch[..., 6], ray_batch[..., 7]  # (N_rays, ) each
    # restore the shape of "near" and "far" to (N_rays, 1)
    near = near[..., None]
    far = far[..., None]
    # extract view directions
    view_dir = ray_batch[..., 8:11]

    t_vals = torch.linspace(0., 1, steps=n_sample)
    if not inv_depth:
        z_vals = near * (1. - t_vals) + far * t_vals
    else:
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * t_vals)

    z_vals = z_vals.expand([n_rays, n_sample])

    if perturb:
        # get intervals between samples
        mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mid, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mid], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        z_vals = lower + (upper - lower) * t_rand

    # sample points for coarse network
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]

    return pts, view_dir, z_vals


def generate_fine_samples(ray_batch, z_vals, weights, n_sample, perturb=True, pytest=False):
    """

    :param ray_batch: Tensor. The compact ray batch returned from generate_ray_batch functions.
                      Must have the shape of (N, 3)
    :param z_vals: array of shape (n_rays, n_coarse_samples). The z values for each coarse sample.
    :param weights: arrays of shape (n_rays, n_coarse_samples). The weights for each coarse sample.
    :param n_sample: int. The number of fine samples to generate.
    :param perturb: bool. If True, each ray is sampled at stratified random points.
    :param pytest:
    :return: pts, view_dir, z_vals.
             pts.shape      = (n_rays, n_coarse_sample + n_fine_sample, 3)
             view_dir.shape = (n_rays, 3)
             z_vals         = (n_rays, n_coarse_sample + n_fine_sample)
    """
    assert ray_batch.shape[-1] == 11

    n_rays = ray_batch.shape[0]
    # extract origin and directions of the rays
    rays_o, rays_d = ray_batch[..., 0:3], ray_batch[..., 3:6]  # (N_rays, 3) each
    # extract view directions
    view_dir = ray_batch[..., 8:11]

    z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], n_sample, det=(not perturb), pytest=pytest)
    z_samples.detach()

    z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples + N_importance, 3]
    return pts, view_dir, z_vals



def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    model_query = NeRF(HiddenDimension=32, device=device)
    model_fine = NeRF(HiddenDimension=32, device=device)
    grad_vars = list(model_query.parameters())
    grad_vars += list(model_fine.parameters())
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    render_kwargs_train = {
        'model_query': model_query,
        'model_fine': model_fine,
        'N_samples' : args.N_samples,
        'N_importance' : args.N_importance,
        'perturb' : args.perturb,
        'raw_noise_std' : args.raw_noise_std
    }

    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, grad_vars, optimizer

def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    for i, c2w in enumerate(render_poses):
        rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], use_viewdirs=True, **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i==0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)


    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps

if __name__ == "__main__":
    from load_blender import load_blender_data
    from argparser import config_parser

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
        exit(0)

    i_train, i_val, i_test = i_split

    # Cast intrinsics to right types
    h, w, focal = hwf
    h = h // 4
    w = w // 4
    focal = focal / 4
    h, w = int(h), int(w)
    hwf = [h, w, focal]

    k = np.array([
        [focal, 0, 0.5 * w],
        [0, focal, 0.5 * h],
        [0, 0, 1]
    ])

    image_train = images[i_train]
    poses_train = poses[i_train]
    N_rand = args.N_rand

    # ray_batch, target_rgb = generate_ray_batch_train(image_train, poses_train, near, far, hwf, k, 1024, 0)

    # print("ray_batch shape:", ray_batch.shape)
    # print("target_rgb shape:", target_rgb.shape)

    # x, view_dir, z_vals = generate_coarse_samples(ray_batch, 64)
    # print("x shape:", x.shape)
    # print("view_dir shape:", view_dir.shape)
    # print("z_vals shape:", z_vals.shape)
    
    render_kwargs_train, render_kwargs_test, grad_vars, optimizer = create_nerf(args)
    
    for i in range(100000):
        print('iteration: ', i)

        # pick random img
        img_i = np.random.choice(i_train)
        target = images[img_i]
        target = torch.Tensor(target).to(device)
        pose = poses[img_i, :3, :4]

        if N_rand is not None:
            rays_o, rays_d = get_rays(h, w, k, torch.Tensor(pose))
            coords = torch.stack(torch.meshgrid(torch.linspace(0, h-1, h), torch.linspace(0, w-1, w)), -1)  # (H, W, 2)

            coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
            select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
            select_coords = coords[select_inds].long()  # (N_rand, 2)
            rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            batch_rays = torch.stack([rays_o, rays_d], 0)
            target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        
        rgb, disp, acc, extras = render(h, w, k, chunk=args.chunk, rays=batch_rays,
                                            verbose=i < 10, retraw=True, use_viewdirs=True, **render_kwargs_train)

        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        trans = extras['raw'][...,-1]
        loss = img_loss
        # psnr = mse2psnr(img_loss)
        loss.backward()
        optimizer.step()

        if i % 2000 == 0:
            print('exporting test scenes')
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test[0:4]]), 
                    hwf, k, args.chunk, render_kwargs_test, gt_imgs=images[i_test], render_factor=4, savedir='./')


