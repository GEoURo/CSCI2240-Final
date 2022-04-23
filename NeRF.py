import math
import torch
import torch.nn as nn
import torch.nn.functional

ReLUGain = math.sqrt(2)


def MSRInitializer(Layer, ActivationGain=1.0):
    FanIn = Layer.weight.data.size(1) * Layer.weight.data[0][0].numel()
    Layer.weight.data.normal_(0, ActivationGain / math.sqrt(FanIn))

    if Layer.bias is not None:
        Layer.bias.data.zero_()

    return Layer


def CreateEmbeddingFunction(L):
    def EmbeddingFunction(x):
        FrequencyRepresentation = [x]  # ??? inconsistent with the paper
        for y in range(L):
            FrequencyRepresentation += [torch.sin(2 ** y * x), torch.cos(2 ** y * x)]
        return torch.cat(FrequencyRepresentation, dim=1)

    return EmbeddingFunction


class NeRF(nn.Module):
    def __init__(self, StemDepth=8, ColorDepth=2,
                 StemHiddenDim=256, ColorHiddenDim=128, GeoFeatDim=256,
                 RequiresPositionEmbedding=(0, 5)):
        super(NeRF, self).__init__()

        LPosition = 10
        LDirection = 4
        PositionEmbeddingDimension = 6 * LPosition + 3  # 6 * LPosition if consistent with the paper
        DirectionEmbeddingDimension = 6 * LDirection + 3  # 6 * LDirection if consistent with the paper

        self.PositionEmbeddingFunction = CreateEmbeddingFunction(LPosition)
        self.DirectionEmbeddingFunction = CreateEmbeddingFunction(LDirection)

        StemLayers = []
        for x in range(StemDepth):
            if x == 0 and x in RequiresPositionEmbedding:
                InputDimension = PositionEmbeddingDimension
                RequiresAuxiliaryInput = False
            elif x in RequiresPositionEmbedding:
                InputDimension = PositionEmbeddingDimension + StemHiddenDim
                RequiresAuxiliaryInput = True
            else:
                InputDimension = StemHiddenDim
                RequiresAuxiliaryInput = False
            StemLayers += [MSRInitializer(nn.Linear(InputDimension, StemHiddenDim), ActivationGain=ReLUGain)]
            StemLayers[-1].RequiresAuxiliaryInput = RequiresAuxiliaryInput

        self.StemLayers = nn.ModuleList(StemLayers)

        self.DensityLayer = MSRInitializer(nn.Linear(StemHiddenDim, 1))
        self.GeoFeatLayer = MSRInitializer(nn.Linear(StemHiddenDim, GeoFeatDim))

        ColorLayers = []
        for i in range(ColorDepth):
            if i == 0:
                InputDimension = DirectionEmbeddingDimension + GeoFeatDim
                OutputDimension = ColorHiddenDim
            elif i == ColorDepth - 1:
                InputDimension = ColorHiddenDim
                OutputDimension = 3
            else:
                InputDimension = ColorHiddenDim
                OutputDimension = ColorHiddenDim

            ColorLayers += [MSRInitializer(nn.Linear(InputDimension, OutputDimension), ActivationGain=ReLUGain)]

        self.ColorLayers = nn.ModuleList(ColorLayers)

    def forward(self, x, d):
        x = self.PositionEmbeddingFunction(x)
        d = self.DirectionEmbeddingFunction(d)

        y = x
        for Layer in self.StemLayers:
            if Layer.RequiresAuxiliaryInput:
                y = Layer(torch.cat([x, y], dim=1))
            else:
                y = Layer(y)
            y = nn.functional.relu(y)

        sigma = self.DensityLayer(y).view(x.shape[0])

        c = torch.cat([self.GeoFeatLayer(y), d], dim=1)

        for i, Layer in enumerate(self.StemLayers):
            if i == len(self.StemLayers) - 1:
                c = Layer(c)
            else:
                c = nn.functional.relu(Layer(c))

        # combine color and sigma into one tensor
        out = torch.cat([c, sigma[:, None]], -1)
        return out
