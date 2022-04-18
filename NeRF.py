import math
import torch
import torch.nn as nn

ReLUGain = math.sqrt(2)

def MSRInitializer(Layer, ActivationGain=1):
    FanIn = Layer.weight.data.size(1) * Layer.weight.data[0][0].numel()
    Layer.weight.data.normal_(0,  ActivationGain / math.sqrt(FanIn))

    if Layer.bias is not None:
        Layer.bias.data.zero_()
    
    return Layer

def CreateEmbeddingFunction(L):
    def EmbeddingFunction(x):
        FrequencyRepresentation = [x] # ??? inconsistent with the paper
        for y in range(L):
            FrequencyRepresentation += [torch.sin(2 ** y * math.pi * x), torch.cos(2 ** y * math.pi * x)]
        return torch.concat(FrequencyRepresentation, dim=1)
    
    return EmbeddingFunction

class NeRF(nn.Module):
    def __init__(self, StemDepth=8, HiddenDimension=256, RequiresPositionEmbedding=[0, 5]):
        super(NeRF, self).__init__()
        
        LPosition = 10
        LDirection = 4
        PositionEmbeddingDimension = 6 * LPosition + 3 # 6 * LPosition if consistent with the paper
        DirectionEmbeddingDimension = 6 * LDirection + 3 # 6 * LDirection if consistent with the paper
        
        StemLayers = []
        for x in range(StemDepth):
            if x == 0 and x in RequiresPositionEmbedding:
                InputDimension = PositionEmbeddingDimension
                RequiresAuxiliaryInput = False
            elif x in RequiresPositionEmbedding:
                InputDimension = PositionEmbeddingDimension + HiddenDimension
                RequiresAuxiliaryInput = True
            else:
                InputDimension = HiddenDimension
                RequiresAuxiliaryInput = False
            StemLayers += [MSRInitializer(nn.Linear(InputDimension, HiddenDimension), ActivationGain=ReLUGain)]
            StemLayers[-1].RequiresAuxiliaryInput = RequiresAuxiliaryInput
        self.StemLayers = nn.ModuleList(StemLayers)
        
        self.DensityLayer = MSRInitializer(nn.Linear(HiddenDimension, 1))
        
        self.RGBLayer1 = MSRInitializer(nn.Linear(HiddenDimension, HiddenDimension))
        self.RGBLayer2 = MSRInitializer(nn.Linear(DirectionEmbeddingDimension + HiddenDimension, HiddenDimension // 2), ActivationGain=ReLUGain)
        self.RGBLayer3 = MSRInitializer(nn.Linear(HiddenDimension // 2, 3))
                
        self.PositionEmbeddingFunction = CreateEmbeddingFunction(LPosition)
        self.DirectionEmbeddingFunction = CreateEmbeddingFunction(LDirection)
        
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
            
        σ = self.DensityLayer(y).view(x.shape[0])
        
        c = torch.concat([self.RGBLayer1(y), d], dim=1)
        c = nn.functional.relu(self.RGBLayer2(c))
        c = self.RGBLayer3(c)
        
        return c, σ
    
    
    
    
    
##### quick test #####    
# x = torch.randn((1024, 3))
# d = torch.randn((1024, 3))
    
# m = NeRF()
# c, σ = m(x, d)

# print(c.shape)
# print(σ.shape)
