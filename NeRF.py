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
        
        self.PrologColorLayer = MSRInitializer(nn.Linear(DirectionEmbeddingDimension + GeoFeatDim, ColorHiddenDim), ActivationGain=ReLUGain)
        ColorLayers = []
        for _ in range(ColorDepth - 2):
            ColorLayers += [MSRInitializer(nn.Linear(ColorHiddenDim, ColorHiddenDim), ActivationGain=ReLUGain)]
        self.ColorLayers = nn.ModuleList(ColorLayers)
        self.EpilogColorLayer = MSRInitializer(nn.Linear(ColorHiddenDim, 3))
            
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

        sigma = self.DensityLayer(y)

        c = torch.cat([self.GeoFeatLayer(y), d], dim=1)
        c = nn.functional.relu(self.PrologColorLayer(c))
        for Layer in self.ColorLayers:
            c = nn.functional.relu(Layer(c))
        c = self.EpilogColorLayer(c)
        
        # combine color and sigma into one tensor
        out = torch.cat([c, sigma], 1)
        return out
    
    
    
    
    
    
    
### Varying Scene NeRF Approach 1: Naive ###
# raw scene parameters p get fed to the NeRF MLP along with x without any transformation
class NaiveNeRF(nn.Module):
    def __init__(self, StemDepth=8, ColorDepth=2,
                 StemHiddenDim=256, ColorHiddenDim=128, GeoFeatDim=256,
                 RequiresPositionEmbedding=(0, 5), SceneParamDim=1):
        super(NaiveNeRF, self).__init__()

        LPosition = 10
        LDirection = 4
        PositionEmbeddingDimension = 6 * LPosition + 3  # 6 * LPosition if consistent with the paper
        DirectionEmbeddingDimension = 6 * LDirection + 3  # 6 * LDirection if consistent with the paper

        self.PositionEmbeddingFunction = CreateEmbeddingFunction(LPosition)
        self.DirectionEmbeddingFunction = CreateEmbeddingFunction(LDirection)
        
        PositionEmbeddingDimension += SceneParamDim

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
        
        self.PrologColorLayer = MSRInitializer(nn.Linear(DirectionEmbeddingDimension + GeoFeatDim, ColorHiddenDim), ActivationGain=ReLUGain)
        ColorLayers = []
        for _ in range(ColorDepth - 2):
            ColorLayers += [MSRInitializer(nn.Linear(ColorHiddenDim, ColorHiddenDim), ActivationGain=ReLUGain)]
        self.ColorLayers = nn.ModuleList(ColorLayers)
        self.EpilogColorLayer = MSRInitializer(nn.Linear(ColorHiddenDim, 3))
            
    def forward(self, x, p, d):
        x = torch.cat([self.PositionEmbeddingFunction(x), p], 1)
        d = self.DirectionEmbeddingFunction(d)
        
        y = x
        for Layer in self.StemLayers:
            if Layer.RequiresAuxiliaryInput:
                y = Layer(torch.cat([x, y], dim=1))
            else:
                y = Layer(y)
            y = nn.functional.relu(y)

        sigma = self.DensityLayer(y)

        c = torch.cat([self.GeoFeatLayer(y), d], dim=1)
        c = nn.functional.relu(self.PrologColorLayer(c))
        for Layer in self.ColorLayers:
            c = nn.functional.relu(Layer(c))
        c = self.EpilogColorLayer(c)
        
        # combine color and sigma into one tensor
        out = torch.cat([c, sigma], 1)
        return out
    
    
    
    
    

    




### Varying Scene NeRF Approach 2: MLP ###
# scene params p get fed to an MLP (ResNet) first, then the latent representation of p gets fed to the NeRF MLP along with x
class BiasedActivation(nn.Module):
    Gain = math.sqrt(2)
    ActivationFunction = nn.functional.silu # generally better than ReLU, normally significantly better for small networks
    
    def __init__(self, InputUnits, ConvolutionalLayer=True):
        super(BiasedActivation, self).__init__()
        
        self.Bias = nn.Parameter(torch.empty(InputUnits))
        self.Bias.data.zero_()
        
        self.ConvolutionalLayer = ConvolutionalLayer
        
    def forward(self, x):
        y = x + self.Bias.view(1, -1, 1, 1) if self.ConvolutionalLayer else x + self.Bias.view(1, -1)
        return BiasedActivation.ActivationFunction(y)

class FullyConnectedBlock(nn.Module):
    def __init__(self, LatentDimension):
        super(FullyConnectedBlock, self).__init__()

        self.LinearLayer1 = MSRInitializer(nn.Linear(LatentDimension, LatentDimension, bias=False), ActivationGain=BiasedActivation.Gain)
        self.LinearLayer2 = MSRInitializer(nn.Linear(LatentDimension, LatentDimension, bias=False), ActivationGain=0)
        
        self.NonLinearity1 = BiasedActivation(LatentDimension, ConvolutionalLayer=False)
        self.NonLinearity2 = BiasedActivation(LatentDimension, ConvolutionalLayer=False)
        
    def forward(self, x):
        y = self.LinearLayer1(self.NonLinearity1(x))
        y = self.LinearLayer2(self.NonLinearity2(y))
        
        return x + y
    
class MLP(nn.Module):
      def __init__(self, SceneParamDim, LatentDimension, Blocks, EnableEpilogNonLinearity):
          super(MLP, self).__init__()
          
          self.PrologLayer = MSRInitializer(nn.Linear(SceneParamDim, LatentDimension, bias=False), ActivationGain=BiasedActivation.Gain)

          self.BlockList = nn.ModuleList([FullyConnectedBlock(LatentDimension) for _ in range(Blocks)])
          
          self.NonLinearity = BiasedActivation(LatentDimension, ConvolutionalLayer=False)
          self.EpilogLayer = MSRInitializer(nn.Linear(LatentDimension, LatentDimension, bias=False), ActivationGain=BiasedActivation.Gain if EnableEpilogNonLinearity else 1)
          
          if EnableEpilogNonLinearity:
              self.EpilogNonLinearity = BiasedActivation(LatentDimension, ConvolutionalLayer=False)
          
      def forward(self, z):
          w = self.PrologLayer(z)
          
          for Block in self.BlockList:
              w = Block(w)
              
          w = self.EpilogLayer(self.NonLinearity(w))
          
          if hasattr(self, 'EpilogNonLinearity'):
              return self.EpilogNonLinearity(w)
          else:
              return w
          
class MLPNeRF(nn.Module):
    def __init__(self, StemDepth=8, ColorDepth=2,
                 StemHiddenDim=256, ColorHiddenDim=128, GeoFeatDim=256,
                 RequiresPositionEmbedding=(0, 5), SceneParamDim=1, SceneParamLatentDim=16, SceneParamMLPDepth=6):
        super(MLPNeRF, self).__init__()

        LPosition = 10
        LDirection = 4
        PositionEmbeddingDimension = 6 * LPosition + 3  # 6 * LPosition if consistent with the paper
        DirectionEmbeddingDimension = 6 * LDirection + 3  # 6 * LDirection if consistent with the paper

        self.PositionEmbeddingFunction = CreateEmbeddingFunction(LPosition)
        self.DirectionEmbeddingFunction = CreateEmbeddingFunction(LDirection)
        
        self.SceneParamMLP = MLP(SceneParamDim, SceneParamLatentDim, SceneParamMLPDepth // 2 - 1, EnableEpilogNonLinearity=False)
        
        PositionEmbeddingDimension += SceneParamLatentDim

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
        
        self.PrologColorLayer = MSRInitializer(nn.Linear(DirectionEmbeddingDimension + GeoFeatDim, ColorHiddenDim), ActivationGain=ReLUGain)
        ColorLayers = []
        for _ in range(ColorDepth - 2):
            ColorLayers += [MSRInitializer(nn.Linear(ColorHiddenDim, ColorHiddenDim), ActivationGain=ReLUGain)]
        self.ColorLayers = nn.ModuleList(ColorLayers)
        self.EpilogColorLayer = MSRInitializer(nn.Linear(ColorHiddenDim, 3))
            
    def forward(self, x, p, d):
        x = torch.cat([self.PositionEmbeddingFunction(x), self.SceneParamMLP(p)], 1)
        d = self.DirectionEmbeddingFunction(d)
        
        y = x
        for Layer in self.StemLayers:
            if Layer.RequiresAuxiliaryInput:
                y = Layer(torch.cat([x, y], dim=1))
            else:
                y = Layer(y)
            y = nn.functional.relu(y)

        sigma = self.DensityLayer(y)

        c = torch.cat([self.GeoFeatLayer(y), d], dim=1)
        c = nn.functional.relu(self.PrologColorLayer(c))
        for Layer in self.ColorLayers:
            c = nn.functional.relu(Layer(c))
        c = self.EpilogColorLayer(c)
        
        # combine color and sigma into one tensor
        out = torch.cat([c, sigma], 1)
        return out
    
    
    
    
    
    
    
    
    
# Daniel's suggestion  
### Varying Scene NeRF Approach 3: Per-Layer Modulation via Mean and STD ###
# scene params p get fed to an MLP first, then the latent representation of p is used to modulate the mean and std of each NeRF layer
class ModulatedFullyConnectedLayer(nn.Module):
    def __init__(self, InputDim, OutputDim, ModulationDim, NormalizeBeforeModulation, ActivationGain):
        super(ModulatedFullyConnectedLayer, self).__init__()
        
        self.MainLayer = MSRInitializer(nn.Linear(InputDim, OutputDim), ActivationGain=ActivationGain)
        
        self.STDModulation = MSRInitializer(nn.Linear(ModulationDim, 1))
        self.STDModulation.bias.data.fill_(1) # from StyleGAN
        
        self.MeanModulation = MSRInitializer(nn.Linear(ModulationDim, 1))
        
        self.NormalizeBeforeModulation = NormalizeBeforeModulation
        
    def forward(self, x, w):
        ModulatedSTD = self.STDModulation(w)
        ModulatedMean = self.MeanModulation(w)
        x = self.MainLayer(x)
        
        if self.NormalizeBeforeModulation:
            STD, Mean = torch.std_mean(x, dim=1, keepdim=True)
            x = (x - Mean) / STD
        
        return ModulatedSTD * x + ModulatedMean
        
class ModulatedNeRF(nn.Module):
    def __init__(self, StemDepth=8, ColorDepth=2,
                 StemHiddenDim=256, ColorHiddenDim=128, GeoFeatDim=256,
                 RequiresPositionEmbedding=(0, 5), SceneParamDim=1, SceneParamLatentDim=16, SceneParamMLPDepth=6, NormalizeBeforeModulation=True):
        super(ModulatedNeRF, self).__init__()

        LPosition = 10
        LDirection = 4
        PositionEmbeddingDimension = 6 * LPosition + 3  # 6 * LPosition if consistent with the paper
        DirectionEmbeddingDimension = 6 * LDirection + 3  # 6 * LDirection if consistent with the paper

        self.PositionEmbeddingFunction = CreateEmbeddingFunction(LPosition)
        self.DirectionEmbeddingFunction = CreateEmbeddingFunction(LDirection)
        
        self.SceneParamMLP = MLP(SceneParamDim, SceneParamLatentDim, SceneParamMLPDepth // 2 - 1, EnableEpilogNonLinearity=True)
        
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
            StemLayers += [ModulatedFullyConnectedLayer(InputDimension, StemHiddenDim, SceneParamLatentDim, NormalizeBeforeModulation, ReLUGain)]
            StemLayers[-1].RequiresAuxiliaryInput = RequiresAuxiliaryInput
        self.StemLayers = nn.ModuleList(StemLayers)

        self.DensityLayer = MSRInitializer(nn.Linear(StemHiddenDim, 1))
        self.GeoFeatLayer = ModulatedFullyConnectedLayer(StemHiddenDim, GeoFeatDim, SceneParamLatentDim, NormalizeBeforeModulation, 1)
        
        self.PrologColorLayer = ModulatedFullyConnectedLayer(DirectionEmbeddingDimension + GeoFeatDim, ColorHiddenDim, SceneParamLatentDim, NormalizeBeforeModulation, ReLUGain)
        ColorLayers = []
        for _ in range(ColorDepth - 2):
            ColorLayers += [ModulatedFullyConnectedLayer(ColorHiddenDim, ColorHiddenDim, SceneParamLatentDim, NormalizeBeforeModulation, ReLUGain)]
        self.ColorLayers = nn.ModuleList(ColorLayers)
        self.EpilogColorLayer = MSRInitializer(nn.Linear(ColorHiddenDim, 3))
            
    def forward(self, x, p, d):
        x = self.PositionEmbeddingFunction(x)
        d = self.DirectionEmbeddingFunction(d)
        w = self.SceneParamMLP(p)
        
        y = x
        for Layer in self.StemLayers:
            if Layer.RequiresAuxiliaryInput:
                y = Layer(torch.cat([x, y], dim=1), w)
            else:
                y = Layer(y, w)
            y = nn.functional.relu(y)

        sigma = self.DensityLayer(y)

        c = torch.cat([self.GeoFeatLayer(y, w), d], dim=1)
        c = nn.functional.relu(self.PrologColorLayer(c, w))
        for Layer in self.ColorLayers:
            c = nn.functional.relu(Layer(c, w))
        c = self.EpilogColorLayer(c)
        
        # combine color and sigma into one tensor
        out = torch.cat([c, sigma], 1)
        return out   
        

        
        

        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
##### quick test #####    
x = torch.randn((1024, 3))
d = torch.randn((1024, 3))
p = torch.randn((1024, 1))
    
m = ModulatedNeRF()
out = m(x, p, d)

print(out.shape)
