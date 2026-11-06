"""DANet-S (port of the authors' repo, crash and head fixed) plus the reference models.

DANet switches, defaults = the released code:
  gate          'softmax' (code, over channels) | 'sigmoid' (paper Eq. 3)
  dilation_mode 'cascade' (code, d1 -> d2 -> d3) | 'parallel' (paper Eq. 1)
  norm          'instance' (code, InstanceNorm2d) | 'layer' (paper "LN")
  attention     'dilated' (code and paper) | 'none' (E13a ablation: each block keeps only its 1x1 projection,
                so the dilated 3x3 convolutions, pooling concat and gate are all removed)
  head_pool     'avg' (code: AvgPool2d(7) over the final 7x7 map) | 'max' (E13b: global max pooling)
The paper does not say which axes its LayerNorm covers. 'layer' uses GroupNorm(1, C): normalisation over
C x H x W with a per-channel affine, the usual conv-net reading. This choice must be stated in the paper.
"""
import torch
import torch.nn as nn
import torchvision

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def make_norm(kind, c):
    if kind == "instance":
        return nn.InstanceNorm2d(c, affine=True)
    if kind == "layer":
        return nn.GroupNorm(1, c)
    raise ValueError(kind)


def conv_norm_act(cin, cout, k, dilation=1, norm="instance"):
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, padding=dilation * (k // 2), dilation=dilation),
        make_norm(norm, cout),
        nn.LeakyReLU(),
    )


class DAB_S(nn.Module):
    """Dilated Attention Block (skip variant)."""
    def __init__(self, cin, cout, gate="softmax", dilation_mode="cascade", norm="instance", attention="dilated"):
        super().__init__()
        assert gate in ("softmax", "sigmoid") and dilation_mode in ("cascade", "parallel")
        assert attention in ("dilated", "none")
        self.gate, self.dilation_mode, self.attention = gate, dilation_mode, attention
        if attention == "none":                                                # E13a: only the projection is left
            self.proj = conv_norm_act(cin, cout, 1, norm=norm)
            return
        self.reduce = nn.Conv2d(cin, cin, 1)                                   # f_c
        self.dilated = nn.ModuleList([conv_norm_act(cin, cin, 3, d, norm) for d in (1, 2, 3)])
        self.avg = nn.AvgPool2d(3, stride=1, padding=1)
        self.max = nn.MaxPool2d(3, stride=1, padding=1)
        self.fuse = conv_norm_act(6 * cin, cin, 3, norm=norm)
        self.att = nn.Conv2d(cin, cin, 1)
        self.proj = conv_norm_act(cin, cout, 1, norm=norm)

    def forward(self, x):
        if self.attention == "none":
            return self.proj(x)
        h, feats = self.reduce(x), []
        for conv in self.dilated:
            src = h if (self.dilation_mode == "parallel" or not feats) else feats[-1]
            feats.append(conv(src))
        f = torch.cat(feats, 1)                                                # f_preConcat (3C)
        f = torch.cat([self.avg(f), self.max(f)], 1)                           # f_poolConcat (6C)
        a = self.att(self.fuse(f))
        a = torch.softmax(a, 1) if self.gate == "softmax" else torch.sigmoid(a)  # f_att
        return self.proj(a * x)


class DANet_S(nn.Module):
    """Expects 224x224 inputs scaled to [0, 1]; returns log-probabilities (B, 2)."""
    def __init__(self, n=8, num_classes=2, head_pool="avg", **dab_kw):
        super().__init__()
        assert head_pool in ("avg", "max")
        self.stem = nn.Sequential(nn.Conv2d(3, n, 3, padding=1), nn.MaxPool2d(2))
        ch = [n, n, 2*n, 2*n, 4*n, 4*n, 8*n, 8*n, 16*n, 16*n]
        layers = []
        for i in range(9):                                   # nine DABs, MaxPool after blocks 1, 3, 5, 7
            layers.append(DAB_S(ch[i], ch[i + 1], **dab_kw))
            if i in (0, 2, 4, 6):
                layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers)               # 112 -> 7 spatial
        self.pool = nn.AvgPool2d(7, stride=2) if head_pool == "avg" else nn.MaxPool2d(7, stride=2)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(16*n, 8*n), nn.ReLU(),
                                  nn.Linear(8*n, num_classes), nn.LogSoftmax(dim=1))

    def embed(self, x):
        return self.pool(self.features(self.stem(x))).flatten(1)

    def forward(self, x):
        return self.head(self.pool(self.features(self.stem(x))))


class ResNet18Ref(nn.Module):
    """ImageNet-pretrained ResNet-18 reference; takes [0, 1] inputs, normalises internally."""
    def __init__(self, pretrained=True, num_classes=2):
        super().__init__()
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.net = torchvision.models.resnet18(weights=w)
        self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def embed(self, x):
        n = self.net
        x = (x - self.mean) / self.std
        x = n.maxpool(n.relu(n.bn1(n.conv1(x))))
        x = n.layer4(n.layer3(n.layer2(n.layer1(x))))
        return torch.flatten(n.avgpool(x), 1)

    def forward(self, x):
        return torch.log_softmax(self.net.fc(self.embed(x)), dim=1)


def build_model(cfg):
    if cfg["model"] == "danet":
        return DANet_S(cfg.get("base_channels", 8), gate=cfg.get("gate", "softmax"),
                       dilation_mode=cfg.get("dilation_mode", "cascade"), norm=cfg.get("norm", "instance"),
                       attention=cfg.get("attention", "dilated"), head_pool=cfg.get("head_pool", "avg"))
    if cfg["model"] == "resnet18":
        return ResNet18Ref(pretrained=cfg.get("pretrained", True))
    raise ValueError(cfg["model"])


def count_params(m):
    return sum(p.numel() for p in m.parameters())
