# FastResUNet (Lightweight Hybrid Segmentation)

## 1. Goal
Perform semantic image segmentation using a highly optimized, lightweight architecture designed to run efficiently on devices with integrated graphics (e.g., Intel Iris Xe) without thermal throttling. This project utilizes the Pascal VOC 2012 dataset.

## 2. Architecture: "FastResUNet"
The model is built for high speed and efficiency, balancing performance with strict hardware constraints.

- **Encoder:** Pre-trained **ResNet-18** backbone (using layers 1 through 4). This acts as a robust feature extractor.
- **Bottleneck (ASPP):** Atrous Spatial Pyramid Pooling.
    - Captures multi-scale contextual information by employing multiple parallel filters with different dilation rates.
    - Features a global average pooling branch to incorporate image-level features.
- **Decoder:** Custom Decoder Blocks with Squeeze-and-Excitation (SE).
    - Utilizes **Skip Connections** from the ResNet encoder to recover fine spatial details and sharp object boundaries.
    - Employs **Group Normalization** and **SE Blocks** to recalibrate channel-wise feature responses adaptively.
- **Output:** A segmentation mask matching the input resolution (e.g., $320 \times 320$), outputting class logits for 21 classes (Pascal VOC).

## 3. Data Augmentation
A comprehensive joint-transformation pipeline is implemented to boost robustness, including:
- Random Scale & Crop (with dynamic padding)
- Geometric Augmentations (Horizontal Flips, Random Rotations)
- Color Jittering (Brightness, Contrast, Saturation, Hue) and Gaussian Blur

## 4. Hardware Constraints & Optimizations
Designed specifically for environments like an HP Pavilion Plus:
- **Processor:** i5-1335U (10 cores).
- **GPU:** Intel Iris Xe (Shared Memory).
- **Strategy:** Uses Group Normalization (better for small batch sizes) instead of Batch Normalization in the decoder, enabling stable training with small batch sizes (4-8) to prevent memory exhaustion and thermal throttling.

## 5. Usage

### Training
Training logic and experiments are conducted within `Train.ipynb`.

### Testing & Visualization
Run the interactive viewer to visualize model predictions alongside ground-truth overlays.
```bash
python test.py
```
*(Use the Left/Right arrow keys to navigate through test images).*