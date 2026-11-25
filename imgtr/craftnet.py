"""CRAFT 글자 찾기 (글자 하나하나 위치를 찾아서 휘어지거나 비스듬한 글씨도 잡음)
원본: CRAFT (Copyright (c) 2019-present NAVER Corp., MIT License) · EasyOCR 판 (Apache-2.0) 을 필요한 부분만 옮김"""
import os
from collections import OrderedDict, namedtuple
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

URL = "https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip"


class _VGG(nn.Module):
    def __init__(self):
        super().__init__()
        f = models.vgg16_bn(weights=None).features
        self.slice1, self.slice2, self.slice3, self.slice4 = (nn.Sequential() for _ in range(4))
        for x in range(12):
            self.slice1.add_module(str(x), f[x])
        for x in range(12, 19):
            self.slice2.add_module(str(x), f[x])
        for x in range(19, 29):
            self.slice3.add_module(str(x), f[x])
        for x in range(29, 39):
            self.slice4.add_module(str(x), f[x])
        self.slice5 = nn.Sequential(nn.MaxPool2d(3, 1, 1), nn.Conv2d(512, 1024, 3, padding=6, dilation=6), nn.Conv2d(1024, 1024, 1))

    def forward(self, x):
        h = self.slice1(x); r2 = h
        h = self.slice2(h); r3 = h
        h = self.slice3(h); r4 = h
        h = self.slice4(h); r5 = h
        h = self.slice5(h)
        return h, r5, r4, r3, r2


class _DC(nn.Module):
    def __init__(self, i, m, o):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(i + m, m, 1), nn.BatchNorm2d(m), nn.ReLU(True),
                                  nn.Conv2d(m, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True))

    def forward(self, x):
        return self.conv(x)


class CRAFT(nn.Module):
    def __init__(self):
        super().__init__()
        self.basenet = _VGG()
        self.upconv1, self.upconv2 = _DC(1024, 512, 256), _DC(512, 256, 128)
        self.upconv3, self.upconv4 = _DC(256, 128, 64), _DC(128, 64, 32)
        self.conv_cls = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(True), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(True),
                                      nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(True), nn.Conv2d(16, 16, 1), nn.ReLU(True), nn.Conv2d(16, 2, 1))

    def forward(self, x):
        s = self.basenet(x)
        y = self.upconv1(torch.cat([s[0], s[1]], 1))
        y = self.upconv2(torch.cat([F.interpolate(y, size=s[2].shape[2:], mode="bilinear", align_corners=False), s[2]], 1))
        y = self.upconv3(torch.cat([F.interpolate(y, size=s[3].shape[2:], mode="bilinear", align_corners=False), s[3]], 1))
        y = self.upconv4(torch.cat([F.interpolate(y, size=s[4].shape[2:], mode="bilinear", align_corners=False), s[4]], 1))
        return self.conv_cls(y).permute(0, 2, 3, 1)


_net = None


def net(cache_dir):
    global _net
    if _net is None:
        import zipfile, io, urllib.request
        p = os.path.join(cache_dir, "craft_mlt_25k.pth")
        if not os.path.exists(p):
            os.makedirs(cache_dir, exist_ok=True)
            data = urllib.request.urlopen(URL, timeout=600).read()
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                with open(p + ".part", "wb") as f:
                    f.write(z.read("craft_mlt_25k.pth"))
            os.replace(p + ".part", p)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        with open(p, "rb") as fh:                          # 경로에 한글이 있어도 열리게 파이썬으로 읽어 넘김
            sd = torch.load(io.BytesIO(fh.read()), map_location=dev, weights_only=False)
        sd = OrderedDict((k[7:] if k.startswith("module.") else k, v) for k, v in sd.items())
        m = CRAFT()
        m.load_state_dict(sd)
        m.to(dev).eval()
        _net = (m, dev)
    return _net


def heat(rgb, cache_dir, long_side=1600):
    """글자 있을 만한 정도 지도 (그림 크기와 같게)"""
    m, dev = net(cache_dir)
    H, W = rgb.shape[:2]
    s = long_side / max(H, W)
    s = min(max(s, 0.5), 2.5)
    w, h = int(W * s) // 32 * 32 or 32, int(H * s) // 32 * 32 or 32
    x = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
    x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    with torch.no_grad():
        y = m(torch.from_numpy(x).permute(2, 0, 1)[None].to(dev))
    sc = y[0, :, :, 0].float().cpu().numpy()
    return cv2.resize(sc, (W, H), interpolation=cv2.INTER_LINEAR)
