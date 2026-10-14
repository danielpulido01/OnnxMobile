"""Arquitecturas manuales: LeNet-5 (1998) y MobileNetV1 (2017)."""
import torch
from torch import nn
from torch.nn import functional as F


class ScaledTanh(nn.Module):
    def forward(self, x):
        return 1.7159 * torch.tanh((2.0 / 3.0) * x)


class Subsampling(nn.Module):
    """Suma 2x2, coeficiente y sesgo entrenables por mapa, sigmoid escalada."""
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AvgPool2d(2, 2, divisor_override=1)
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.25))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.activation = ScaledTanh()

    def forward(self, x):
        return self.activation(self.pool(x) * self.scale + self.bias)


class PartialC3(nn.Module):
    # Tabla I, LeCun et al. (1998): 60 kernels y 16 sesgos = 1516 parámetros.
    connections = (
        (0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 5), (0, 4, 5), (0, 1, 5),
        (0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5), (0, 3, 4, 5),
        (0, 1, 4, 5), (0, 1, 2, 5), (0, 1, 3, 4), (1, 2, 4, 5),
        (0, 2, 3, 5), (0, 1, 2, 3, 4, 5),
    )

    def __init__(self):
        super().__init__()
        self.maps = nn.ModuleList(nn.Conv2d(len(c), 1, 5) for c in self.connections)

    def forward(self, x):
        return torch.cat([layer(x[:, list(c)]) for layer, c in zip(self.maps, self.connections)], dim=1)


class EuclideanRBF(nn.Module):
    """Prototipos fijos ±1; logits = distancia euclidiana al cuadrado negativa.

    Los códigos de glifos de dígitos del artículo no aplican a palabras.
    Se reemplazan por códigos binarios deterministas para las clases de audio.
    Esta adaptación se documenta en el notebook 02_entrenamiento_evaluacion.ipynb.
    """
    def __init__(self, classes):
        super().__init__()
        generator = torch.Generator().manual_seed(1998)
        codes = torch.randint(0, 2, (classes, 84), generator=generator).float() * 2 - 1
        if len(torch.unique(codes, dim=0)) != classes:
            raise ValueError("Prototipos RBF repetidos")
        self.register_buffer("codes", codes)

    def forward(self, x):
        return -((x[:, None, :] - self.codes[None, :, :]) ** 2).sum(dim=2)


class LeNet5(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.c1 = nn.Conv2d(1, 6, 5)
        self.s2 = Subsampling(6)
        self.c3 = PartialC3()
        self.s4 = Subsampling(16)
        self.c5 = nn.Conv2d(16, 120, 5)
        self.f6 = nn.Linear(120, 84)
        self.output = EuclideanRBF(classes)
        self.activation = ScaledTanh()

    def forward(self, x):
        # Adaptación de entrada: conserva el contrato público NCHW 64x101.
        x = F.interpolate(x, size=(32, 32), mode="bilinear", align_corners=False)
        x = self.s2(self.activation(self.c1(x)))
        x = self.s4(self.activation(self.c3(x)))
        x = self.activation(self.c5(x)).flatten(1)
        return self.output(self.activation(self.f6(x)))


class ConvBNReLU(nn.Sequential):
    def __init__(self, inputs, outputs, kernel=3, stride=1, groups=1):
        super().__init__(
            nn.Conv2d(inputs, outputs, kernel, stride, kernel // 2, groups=groups, bias=False),
            nn.BatchNorm2d(outputs), nn.ReLU(inplace=False),
        )


class DepthwiseSeparable(nn.Sequential):
    def __init__(self, inputs, outputs, stride):
        super().__init__(ConvBNReLU(inputs, inputs, stride=stride, groups=inputs),
                         ConvBNReLU(inputs, outputs, kernel=1))


class MobileNetV1(nn.Module):
    def __init__(self, classes, width_mult=0.25):
        super().__init__()
        if not 0 < width_mult <= 1:
            raise ValueError("width_mult debe estar en (0, 1]")
        width = lambda c: max(1, int(c * width_mult))
        blocks = [(64, 1), (128, 2), (128, 1), (256, 2), (256, 1),
                  (512, 2)] + [(512, 1)] * 5 + [(1024, 2), (1024, 1)]
        layers = [ConvBNReLU(1, width(32), stride=2)]
        inputs = width(32)
        for outputs, stride in blocks:
            layers.append(DepthwiseSeparable(inputs, width(outputs), stride))
            inputs = width(outputs)
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(inputs, classes)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)).flatten(1))


def build_model(architecture, classes, width_mult=0.25):
    if architecture == "lenet5":
        return LeNet5(classes)
    if architecture == "mobilenet_v1":
        return MobileNetV1(classes, width_mult)
    raise ValueError(f"Arquitectura desconocida: {architecture}")
