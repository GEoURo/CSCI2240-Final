import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NeRF(nn.Module):
    def __init__(self, num_layers=8, hidden_dim=256,
                 input_dim_pts=3, input_dim_view=3, output_dim=4,
                 skips=None):
        """
        Unlike the official NeRF implementation, our implementation by default consider using view direction
        :param num_layers:
        :param hidden_dim:
        :param input_dim_pts:
        :param input_dim_view:
        :param output_dim:
        :param skips: default to [4] if remain as None.
        """
        super(NeRF, self).__init__()

        if skips is None:
            skips = [4]

        # setup member variables
        self.num_layers = num_layers
        self.hidden_layer_dim = hidden_dim
        self.input_dim_pts = input_dim_pts
        self.input_dim_view = input_dim_view
        self.output_dim = output_dim
        self.skips = skips

        pos_layers = [nn.Linear(input_dim_pts, hidden_dim)]
        for i in range(num_layers - 1):
            if i in self.skips:
                # for skip layers, the direct position input will be concatenated to the input
                pos_layers.append(nn.Linear(input_dim_pts + hidden_dim, hidden_dim))
            else:
                pos_layers.append(nn.Linear(hidden_dim, hidden_dim))

        # create position network
        self.pos_layers = nn.ModuleList(pos_layers)

        # create color network
        self.view_layers = nn.ModuleList([nn.Linear(input_dim_view + hidden_dim, hidden_dim // 2)])

        self.feature_layer = nn.Linear(hidden_dim, hidden_dim)
        self.density_layer = nn.Linear(hidden_dim, 1)
        self.color_layer = nn.Linear(hidden_dim // 2, 3)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_dim_pts, self.input_dim_view], dim=-1)
        h = input_pts
        # forward pass of the position network
        for i, layer in enumerate(self.pos_layers):
            h = self.pos_layers[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], dim=-1)

        # h: (..., hidden_layer_dim)
        # forward pass of the color network

        # compute density directly from position inputs
        sigma = self.density_layer(h)
        # compute feature for the position and concatenate with view direction (encoding)
        feature = self.feature_layer(h)
        h = torch.cat([feature, input_views], dim=-1)
        for i, layer in enumerate(self.view_layers):
            h = self.view_layers[i](h)
            h = F.relu(h)
        # compute rgb prediction
        rgb = self.color_layer(h)
        outputs = torch.cat([rgb, sigma], dim=-1)

        return outputs


class NeRFBaby(nn.Module):
    def __init__(self, num_layers_pts=3, num_layers_colors=4,
                 hidden_dim=64, view_feat_dim=15,
                 input_dim_pts=3, input_dim_view=3):
        """
        :param num_layers_pts: number of layers for the position net, default to 3
        :param num_layers_colors: number of layers for the color net, default to 4
        :param hidden_dim: dimension for all hidden layers, default to 64
        :param view_feat_dim: default to 15 for spherical harmonics
        :param input_dim_pts: dimension for position input, default to 3
        :param input_dim_view: dimension for view direction input, default to 3
        """
        super(NeRFBaby, self).__init__()

        self.input_dim_pts = input_dim_pts
        self.input_dim_view = input_dim_view

        self.num_layers_pts = num_layers_pts
        self.num_layers_colors = num_layers_colors
        self.hidden_dim = hidden_dim
        self.view_feat_dim = view_feat_dim

        # create position network
        pos_layers = []
        for i in range(num_layers_pts):
            if i == 0:
                in_dim = self.input_dim_pts
            else:
                in_dim = self.hidden_dim

            if i == num_layers_pts - 1:
                # (1 density sigma + 15 spherical harmonics) features for color net input
                out_dim = 1 + self.view_feat_dim
            else:
                out_dim = self.hidden_dim

            pos_layers.append(nn.Linear(in_dim, out_dim, bias=False))

        self.pos_layers = nn.ModuleList(pos_layers)

        # create color network
        color_layers = []
        for i in range(num_layers_colors):
            if i == 0:
                in_dim = self.input_dim_view + self.view_feat_dim
            else:
                in_dim = self.hidden_dim

            if i == num_layers_colors - 1:
                out_dim = 3
            else:
                out_dim = self.hidden_dim

            color_layers.append(nn.Linear(in_dim, out_dim, bias=False))

        self.color_layers = nn.ModuleList(color_layers)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_dim_pts, self.input_dim_view], dim=-1)
        h = input_pts
        # forward pass of the position network
        for i in range(self.num_layers_pts):
            h = self.pos_layers[i](h)
            if i != self.num_layers_pts - 1:
                h = F.relu(h, inplace=True)

        # split sigma and view feature
        sigma, view_feature = h[..., 0], h[..., 1:]

        h = torch.cat([input_views, view_feature], dim=-1)
        # forward pass of the color network
        for i in range(self.num_layers_colors):
            h = self.color_layers[i](h)
            if i != self.num_layers_colors - 1:
                h = F.relu(h, inplace=True)

        color = h
        outputs = torch.cat([color, sigma.unsqueeze(dim=-1)], dim=-1)
        return outputs
