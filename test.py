import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, models
from PIL import Image

# ==========================================
# 1. --- ARCHITECTURE DEFINITIONS (UPDATED)
# ==========================================

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False)

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.ReLU(inplace=True)
        )

        # ✅ FIXED: 4 branches
        self.bn = nn.GroupNorm(32, out_channels * 4)
        self.relu = nn.ReLU(inplace=True)

        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, 1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        x4 = self.global_pool(x)
        x4 = F.interpolate(x4, size=x.shape[-2:], mode='bilinear', align_corners=True)

        out = torch.cat([x1, x2, x3, x4], dim=1)
        out = self.relu(self.bn(out))
        return self.project(out)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):  # ⚠️ must match training
        super().__init__()

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(16, out_channels),
            nn.ReLU(inplace=True)
        )

        self.se = SEBlock(out_channels)

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(16, out_channels),
            nn.ReLU(inplace=True)
        )

        # ✅ residual (important for matching weights)
        self.residual = nn.Conv2d(in_channels + skip_channels, out_channels, 1, bias=False)

    def forward(self, x, skip):
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=True)

        x_cat = torch.cat([x, skip], dim=1)

        out = self.conv1(x_cat)
        out = self.se(out)
        out = self.conv2(out)

        res = self.residual(x_cat)

        return out + res


class FastResUNet(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()

        resnet = models.resnet18(weights='IMAGENET1K_V1')

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.aspp = ASPP(512, 256)

        self.up1 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.dec1 = DecoderBlock(256, 256, 128)

        self.up2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.dec2 = DecoderBlock(128, 128, 64)

        self.up3 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec3 = DecoderBlock(64, 64, 64)

        self.up4 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec4 = DecoderBlock(32, 64, 32)

        # ✅ matches training
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1)
        )

    def forward(self, x):
        l0 = self.layer0(x)
        l1 = self.layer1(self.pool(l0))
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        bottleneck = self.aspp(l4)

        d1 = self.dec1(self.up1(bottleneck), l3)
        d2 = self.dec2(self.up2(d1), l2)
        d3 = self.dec3(self.up3(d2), l1)
        d4 = self.dec4(self.up4(d3), l0)

        return self.final(d4)


# ==========================================
# 2. --- VISUALIZATION SCRIPT (UNCHANGED)
# ==========================================

def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def interactive_viewer(image_folder, model_path, device="cpu", img_size=(320, 320), apply_mc_dropout=False):

    print(f"Loading model from {model_path} onto {device}...")

    model = FastResUNet(num_classes=21).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    image_paths = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))
    if not image_paths:
        print(f"Error: No .jpg images found in {image_folder}")
        return

    print(f"Found {len(image_paths)} images. Use Left/Right arrow keys to navigate.")

    state = {'idx': 0}
    num_plots = 4 if apply_mc_dropout else 3

    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))
    plt.subplots_adjust(bottom=0.15)

    def update_plot():
        for ax in axes:
            ax.clear()
            ax.axis('off')

        img_path = image_paths[state['idx']]
        img_pil = Image.open(img_path).convert("RGB")
        img_resized = img_pil.resize(img_size)

        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        input_tensor = transform(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        axes[0].imshow(img_resized)
        axes[0].set_title("Original")

        axes[1].imshow(pred_mask, cmap='tab20', vmin=0, vmax=20)
        axes[1].set_title("Mask")

        axes[2].imshow(img_resized)
        axes[2].imshow(pred_mask, cmap='tab20', alpha=0.5)
        axes[2].set_title("Overlay")

        fig.canvas.draw_idle()

    def on_key_press(event):
        if event.key == 'right':
            state['idx'] = (state['idx'] + 1) % len(image_paths)
            update_plot()
        elif event.key == 'left':
            state['idx'] = (state['idx'] - 1) % len(image_paths)
            update_plot()

    fig.canvas.mpl_connect('key_press_event', on_key_press)
    update_plot()
    plt.show()


# ==========================================
# RUN
# ==========================================

if __name__ == "__main__":
    device = torch.device("cpu")

    interactive_viewer(
        image_folder="./Pascal_VOC_2012/VOC2012_test/VOC2012_test/JPEGImages/",
        model_path="./best_fast_model.pth",
        device=device,
        apply_mc_dropout=False
    )