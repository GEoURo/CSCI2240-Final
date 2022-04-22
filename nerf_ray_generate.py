from nerf_utils import *


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
