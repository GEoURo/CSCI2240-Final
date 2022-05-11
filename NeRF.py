import math
import torch
import torch.nn as nn
import torch.nn.functional
from hash_encoder import INGPHashEncoder, SHEncoder

ReLUGain = math.sqrt(2)


def MSRInitializer(Layer, ActivationGain=1.0):
    FanIn = Layer.weight.data.size(1) * Layer.weight.data[0][0].numel()
    Layer.weight.data.normal_(0, ActivationGain / math.sqrt(FanIn))

    if Layer.bias is not None:
        Layer.bias.data.zero_()

    return Layer


def CreateEmbedding(EmbeddingType, L=10,
                    BoundingBox=None, Log2TableSize=19, FinestRes=512):
    if EmbeddingType == "hash":
        HashTable = INGPHashEncoder(bounding_box=BoundingBox,
                                    log2_table_size=Log2TableSize,
                                    finest_resolution=FinestRes)
        return HashTable, HashTable.output_dim

    elif EmbeddingType == "spherical":
        SphericalHarmonics = SHEncoder()
        return SphericalHarmonics, SphericalHarmonics.out_dim

    elif EmbeddingType == "pos":
        def EmbeddingFunction(x):
            FrequencyRepresentation = [x]  # ??? inconsistent with the paper
            for y in range(L):
                FrequencyRepresentation += [torch.sin(2 ** y * x), torch.cos(2 ** y * x)]
            return torch.cat(FrequencyRepresentation, dim=1)

        return EmbeddingFunction, 6 * L + 3

    return nn.Identity, 3


class NeRF(nn.Module):
    def __init__(self, StemDepth=8, ColorDepth=2,
                 StemHiddenDim=256, ColorHiddenDim=128, GeoFeatDim=256,
                 RequiresPositionEmbedding=(0, 5), INGP=False,
                 BoundingBox=None, Log2TableSize=19, FinestRes=512, nAuxParams=0):
        """
        :param StemDepth: int. The number of layers for position network
        :param ColorDepth: int. The number of layers for color network
        :param StemHiddenDim: int. The size of hidden dimension for position network
        :param ColorHiddenDim: int. The size of hidden dimension for color network
        :param GeoFeatDim: int. The dimension of geometry feature.
            The feature will be first concatenated with view direction embedding
            to serve as the input of the color network
        :param RequiresPositionEmbedding: tuple. The layers in the position network
            that requires position embedding as input.
        :param INGP: bool. Enable instant neural graphic primitive
        :param BoundingBox: array of shape [2, 3]. the bounding box of the scene.
        :param Log2TableSize: int. log2(TableSize), default to 19.
        :param FinestRes: int. Finest resolution of the hash table, default to 512.
        """
        super(NeRF, self).__init__()

        self.__INGP = INGP

        # create position and direction encoding
        if INGP:
            self.PositionEmbedding, PositionEmbeddingDim = CreateEmbedding(EmbeddingType="hash",
                                                                           BoundingBox=BoundingBox,
                                                                           Log2TableSize=Log2TableSize,
                                                                           FinestRes=FinestRes)

            self.DirectionEmbedding, DirectionEmbeddingDim = CreateEmbedding(EmbeddingType="spherical")
        else:
            self.PositionEmbedding, PositionEmbeddingDim = CreateEmbedding(EmbeddingType="pos", L=10)
            self.DirectionEmbedding, DirectionEmbeddingDim = CreateEmbedding(EmbeddingType="pos", L=4)
        
        self.LightPosEmbedding, LightPosEmbeddingDim = CreateEmbedding(EmbeddingType='pos', L=4)

        StemLayers = []
        for x in range(StemDepth):
            if x == 0 and x in RequiresPositionEmbedding:
                InputDimension = PositionEmbeddingDim
                # InputDimension += nAuxParams
                InputDimension += LightPosEmbeddingDim
                RequiresAuxiliaryInput = False
            elif x in RequiresPositionEmbedding:
                InputDimension = PositionEmbeddingDim + StemHiddenDim
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
                InputDimension = DirectionEmbeddingDim + GeoFeatDim
                # InputDimension += nAuxParams
                # InputDimension += LightPosEmbeddingDim
                OutputDimension = ColorHiddenDim
            elif i == ColorDepth - 1:
                InputDimension = ColorHiddenDim
                OutputDimension = 3
            else:
                InputDimension = ColorHiddenDim
                OutputDimension = ColorHiddenDim

            ColorLayers += [MSRInitializer(nn.Linear(InputDimension, OutputDimension), ActivationGain=ReLUGain)]

        self.ColorLayers = nn.ModuleList(ColorLayers)

    def forward(self, x, d, p=torch.zeros(0)):
        # print(p.shape)
        # print(p)
        # if p.shape[0] == 0:
        #     print('NO SCENE PARAMS')
        p = p.unsqueeze(0).expand(x.shape[0], -1)
        p = self.LightPosEmbedding(p)
        # p = self.PositionEmbedding(p)

        x = self.PositionEmbedding(x)
        # print(x.shape)
        # 0/0
        # x = torch.cat([x, p], dim=1)
        x = torch.cat([x, p], dim=1)
        d = self.DirectionEmbedding(d)

        y = x
        for Layer in self.StemLayers:
            if Layer.RequiresAuxiliaryInput:
                y = Layer(torch.cat([x, y], dim=1))
            else:
                y = Layer(y)
            y = nn.functional.relu(y)

        sigma = self.DensityLayer(y).view(x.shape[0])

        c = torch.cat([self.GeoFeatLayer(y), d], dim=1)

        for i, Layer in enumerate(self.ColorLayers):
            if i == len(self.ColorLayers) - 1:
                c = Layer(c)
            else:
                c = nn.functional.relu(Layer(c))

        # combine color and sigma into one tensor
        out = torch.cat([c, sigma[:, None]], -1)
        return out

    def isINGP(self):
        return self.__INGP