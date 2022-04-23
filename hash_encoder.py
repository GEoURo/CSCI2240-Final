import torch
import torch.nn as nn
import utils


class INGPHashEncoder(nn.Module):
    def __init__(self, bounding_box, n_levels=16, n_feature_per_level=2,
                 log2_table_size=19, coarsest_resolution=16, finest_resolution=512):
        """
        bounding_box: array of 2 * 3, the bounding box of the scene
        """
        super(INGPHashEncoder, self).__init__()
        self.bounding_box = bounding_box
        self.n_levels = n_levels
        self.n_feature_per_level = n_feature_per_level
        self.log2_table_size = log2_table_size
        # convert to torch tensor
        self.coarsest_resolution = torch.tensor(coarsest_resolution)
        self.finest_resolution = torch.tensor(finest_resolution)
        # compute output dimension
        self.output_dim = self.n_levels * self.n_feature_per_level
        # setup b
        self.b = torch.exp((torch.log(self.finest_resolution) - torch.log(self.coarsest_resolution)) / (n_levels - 1))
        # setup hash table
        num_embeddings = 2 ** log2_table_size
        self.embeddings = nn.ModuleList([nn.Embedding(num_embeddings, n_feature_per_level) for _ in range(n_levels)])

        # initialize weight with uniform distribution
        for i in range(n_levels):
            nn.init.uniform_(self.embeddings[i].weight, a=-0.0001, b=0.0001)

        return

    def forward(self, x):
        """
        param x: 3D point positions, with the shape of (N, 3)
        return: The shape of the output will be (N, self.output_dim)
        """
        x_all_embedding = []
        # for each level of hash table
        for i in range(self.n_levels):
            resolution = torch.floor(self.coarsest_resolution * (self.b ** i))
            voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices = self.get_voxel_vertices(x, resolution)
            # obtain embedding from the hash table, (N, 8, n_feature_per_level)
            voxel_embedding = self.embeddings[i](hashed_voxel_indices)
            # perform trilinear interpolation, (N, n_feature_per_level)
            x_embedding = self.trilinear_interpolate(x, voxel_min_vertex, voxel_max_vertex, voxel_embedding)
            # accumulate the output
            x_all_embedding.append(x_embedding)

        # concatenate the output from all levels together to get the final output, (N, self.output_dim)
        return torch.cat(x_all_embedding, dim=-1)

    def get_voxel_vertices(self, x, resolution):
        """
        param x: 3D point positions, with the shape of (N, 3)
        param resolution: number of voxel per axis
        return: (N, 3), (N, 3), (N, 8)
        """
        box_min, box_max = self.bounding_box

        if not torch.all(x <= box_max) or not torch.all(x >= box_min):
            print("clamp points because some of them are out of bounds")
            x = torch.clamp(x, min=box_min, max=box_max)

        grid_size = (box_max - box_min) / resolution

        # compute the index of the bottom left vertex of each point's corresponding voxel
        bottom_left_index = torch.floor((x - box_min) / grid_size).int()
        # compute the minimum vertex position for each point's corresponding voxel
        voxel_min_vertex = bottom_left_index * grid_size + box_min
        # we can compute the maximum vertex position by just adding the size of the voxel
        voxel_max_vertex = voxel_min_vertex + torch.tensor([1.0, 1.0, 1.0]) * grid_size

        # bottom_left_index: (N, 3), BOX_OFFSET: (1, 8, 3)
        # we want to broadcast 8 offsets to each index, hence we expand bottom_left_index to (N, 1, 3)
        # to get the shape of (N, 8, 3)
        voxel_indices = bottom_left_index.unsqueeze(1) + utils.BOX_OFFSETS
        # compute hash indices
        hashed_voxel_indices = self.hashed_indices(voxel_indices)

        return voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices

    def hashed_indices(self, x):
        primes = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]

        xor_result = torch.zeros_like(x)[..., 0]
        for i in range(x.shape[-1]):
            xor_result ^= x[..., i] * primes[i]

        return torch.tensor((1 << self.log2_table_size) - 1).to(xor_result.device) & xor_result

    @staticmethod
    def trilinear_interpolate(x, voxel_min_vertex, voxel_max_vertex, voxel_embedding):
        """
        param x: N * 3,
        param voxel_min_vertex: The minimum vertex position of each x's corresponding voxel, with the shape of (N, 3)
        param voxel_max_vertex: The maximum vertex position of each x's corresponding voxel, with the shape of (N, 3)
        param voxel_embedding: The embedding of each x's corresponding voxel, with the shape of (N, 8, 2)
        return: (N, 2)
        """
        # https://en.wikipedia.org/wiki/Trilinear_interpolation
        weights = (x - voxel_min_vertex) / (voxel_max_vertex - voxel_min_vertex)  # N x 3

        # weights.shape = (N, 3), therefore, weights[:, 0] will have shape (N, )
        # because we want to perform elementwise multiplication, we need to make
        # sure its dimension matches with that of voxel_embedding[:, 0], which is (N, 2)
        # Hence, we need to expand the dimension of weights[:, 0] to (N, 1) using weights[:, 0][:, None]

        # step 1
        # 0->000, 1->001, 2->010, 3->011, 4->100, 5->101, 6->110, 7->111
        c00 = voxel_embedding[:, 0] * (1 - weights[:, 0][:, None]) + voxel_embedding[:, 4] * weights[:, 0][:, None]
        c01 = voxel_embedding[:, 1] * (1 - weights[:, 0][:, None]) + voxel_embedding[:, 5] * weights[:, 0][:, None]
        c10 = voxel_embedding[:, 2] * (1 - weights[:, 0][:, None]) + voxel_embedding[:, 6] * weights[:, 0][:, None]
        c11 = voxel_embedding[:, 3] * (1 - weights[:, 0][:, None]) + voxel_embedding[:, 7] * weights[:, 0][:, None]

        # step 2
        c0 = c00 * (1 - weights[:, 1][:, None]) + c10 * weights[:, 1][:, None]
        c1 = c01 * (1 - weights[:, 1][:, None]) + c11 * weights[:, 1][:, None]

        # step 3
        c = c0 * (1 - weights[:, 2][:, None]) + c1 * weights[:, 2][:, None]

        return c


class SHEncoder(nn.Module):
    def __init__(self, input_dim=3, degree=4):

        super().__init__()

        self.input_dim = input_dim
        self.degree = degree

        assert self.input_dim == 3
        assert self.degree >= 1 and self.degree <= 5

        self.out_dim = degree ** 2

        self.C0 = 0.28209479177387814
        self.C1 = 0.4886025119029199
        self.C2 = [
            1.0925484305920792,
            -1.0925484305920792,
            0.31539156525252005,
            -1.0925484305920792,
            0.5462742152960396
        ]
        self.C3 = [
            -0.5900435899266435,
            2.890611442640554,
            -0.4570457994644658,
            0.3731763325901154,
            -0.4570457994644658,
            1.445305721320277,
            -0.5900435899266435
        ]
        self.C4 = [
            2.5033429417967046,
            -1.7701307697799304,
            0.9461746957575601,
            -0.6690465435572892,
            0.10578554691520431,
            -0.6690465435572892,
            0.47308734787878004,
            -1.7701307697799304,
            0.6258357354491761
        ]

    def forward(self, input, **kwargs):

        result = torch.empty((*input.shape[:-1], self.out_dim), dtype=input.dtype, device=input.device)
        x, y, z = input.unbind(-1)

        result[..., 0] = self.C0
        if self.degree > 1:
            result[..., 1] = -self.C1 * y
            result[..., 2] = self.C1 * z
            result[..., 3] = -self.C1 * x
            if self.degree > 2:
                xx, yy, zz = x * x, y * y, z * z
                xy, yz, xz = x * y, y * z, x * z
                result[..., 4] = self.C2[0] * xy
                result[..., 5] = self.C2[1] * yz
                result[..., 6] = self.C2[2] * (2.0 * zz - xx - yy)
                # result[..., 6] = self.C2[2] * (3.0 * zz - 1) # xx + yy + zz == 1, but this will lead to different backward gradients, interesting...
                result[..., 7] = self.C2[3] * xz
                result[..., 8] = self.C2[4] * (xx - yy)
                if self.degree > 3:
                    result[..., 9] = self.C3[0] * y * (3 * xx - yy)
                    result[..., 10] = self.C3[1] * xy * z
                    result[..., 11] = self.C3[2] * y * (4 * zz - xx - yy)
                    result[..., 12] = self.C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
                    result[..., 13] = self.C3[4] * x * (4 * zz - xx - yy)
                    result[..., 14] = self.C3[5] * z * (xx - yy)
                    result[..., 15] = self.C3[6] * x * (xx - 3 * yy)
                    if self.degree > 4:
                        result[..., 16] = self.C4[0] * xy * (xx - yy)
                        result[..., 17] = self.C4[1] * yz * (3 * xx - yy)
                        result[..., 18] = self.C4[2] * xy * (7 * zz - 1)
                        result[..., 19] = self.C4[3] * yz * (7 * zz - 3)
                        result[..., 20] = self.C4[4] * (zz * (35 * zz - 30) + 3)
                        result[..., 21] = self.C4[5] * xz * (7 * zz - 3)
                        result[..., 22] = self.C4[6] * (xx - yy) * (7 * zz - 1)
                        result[..., 23] = self.C4[7] * xz * (xx - 3 * yy)
                        result[..., 24] = self.C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))

        return result
