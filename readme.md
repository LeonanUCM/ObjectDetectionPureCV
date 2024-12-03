# Object Detection and Analysis Using Lightweight Computer Vision Techniques

## Introduction
Object detection and segmentation in images are essential tasks across various domains, including agriculture, industrial quality control, and surveillance. While state-of-the-art solutions often rely on machine learning models such as Convolutional Neural Networks (CNNs) or YOLO (You Only Look Once), they demand significant computational resources and specialized hardware like GPUs.

In environments where such resources are unavailable, there is a need for a robust alternative that leverages classical computer vision techniques. This project addresses these constraints by providing a lightweight pipeline designed to operate efficiently on devices with limited computational power, using only standard Python libraries like OpenCV and NumPy.

---

## Problem Statement
The challenges addressed by this project include:
1. **Hardware Limitations**: Lack of access to high-performance GPUs or modern CPUs.
2. **Restricted Software Environments**: Inability to run advanced machine learning frameworks.
3. **Scalability and Efficiency**: Requirement for a scalable solution that can handle a large number of images in a batch while allowing fine-tuned control when needed.
4. **Versatility**: Need for a system that works across various applications, such as fruit detection in agriculture, quality control in manufacturing, and object tracking in surveillance.

---

## Proposed Solution
This solution implements a highly efficient pipeline for object detection and analysis using classical computer vision methods. The design ensures adaptability across different scenarios while maintaining lightweight performance. The system:
- **Leverages OpenCV** for image processing tasks.
- **Uses Profiles** to optimize parameter settings for specific use cases (e.g., different fruit types).
- **Supports Both Batch and Interactive Modes**, catering to large-scale automation and manual parameter tuning.

---

## Key Features of the Solution

### 1. Efficient Object Detection
- Identifies objects based on geometric and morphological criteria.
- Detects circular objects using `cv2.SimpleBlobDetector` and `cv2.HoughCircles`.

### 2. Configurable Modes
- **Batch Mode** (`BATCH_MODE=True`): Processes entire directories of images automatically.
- **Interactive Mode** (`BATCH_MODE=False`): Allows real-time parameter adjustment and debugging.

### 3. Lightweight Design
- Compatible with limited hardware and software environments.
- Uses standard libraries such as OpenCV and NumPy.

### 4. Profiles for Adaptability
- Profiles are dynamic configurations stored externally, allowing adjustments for various use cases.
- Tuning mode helps refine and save settings into profiles for reuse.

---

## How It Works

### Preprocessing
- **Contrast Enhancement**: Uses CLAHE (Contrast Limited Adaptive Histogram Equalization) to improve visibility.
- **Noise Reduction**: Applies Gaussian and median filtering to clean images while preserving essential features.

### Segmentation
- **Color Filtering**: Converts images to HSV or LAB color spaces for targeted object isolation.
- **Morphological Operations**: Refines object boundaries and removes noise using dilation, erosion, opening, and closing.

### Object Detection
- **Contour and Blob Detection**: Identifies objects based on shape and geometric properties.
- **Circle Detection**: Uses the Hough Circle Transform for precise circular object identification.
- **K-Means Clustering**: Groups pixel intensities to segment images based on dominant features.

### Parameter Adjustment
- Provides interactive sliders for fine-tuning thresholds, kernel sizes, and detection parameters.
- Saves configurations to profiles for automated processing in batch mode.

### Output
- Saves annotated images with detected objects.
- Generates logs containing object counts, sizes, and positions for further analysis.

---

## Profiles and Tuning
Profiles are external configuration files used to adjust the pipeline’s parameters dynamically. These profiles optimize the system for specific object types or image conditions. The `load_config` function facilitates loading and managing these profiles.

### Example Profiles
- **Oranges**: Focuses on detecting medium-sized fruits with specific hue ranges.
- **Apples**: Enhances green and red hues for leaf and fruit differentiation.
- **Yellow Peaches**: Emphasizes texture removal and sensitivity to lighter hues.
- **Red Peaches**: Similar to yellow peaches but emphasizes red hues.

### Tuning Mode
- Adjust parameters interactively using sliders.
- Save configurations directly into profile files for subsequent reuse.

---

## Key Computer Vision Methods Used

| **Functionality**        | **OpenCV Methods**                                  |
|---------------------------|----------------------------------------------------|
| Image Resizing            | `cv2.resize`                                       |
| Color Space Conversion    | `cv2.cvtColor`                                     |
| Histogram Equalization    | `cv2.createCLAHE`                                  |
| Noise Reduction           | `cv2.medianBlur`, `cv2.GaussianBlur`               |
| Morphological Operations  | `cv2.morphologyEx` (dilation, erosion, opening, closing) |
| Circle Detection          | `cv2.HoughCircles`                                 |
| Blob Detection            | `cv2.SimpleBlobDetector`                           |
| Thresholding              | `cv2.threshold`                                    |
| K-Means Clustering        | `cv2.kmeans`                                       |
| Contour Detection         | `cv2.findContours`                                 |
| Heatmap Generation        | `cv2.applyColorMap`                                |

---

## Applications

### 1. Agriculture
- **Example**: Counting oranges in drone-captured images for yield estimation.

### 2. Industrial Quality Control
- **Example**: Detecting shape anomalies or surface defects in manufactured items.

### 3. Surveillance and Monitoring
- **Example**: Identifying circular shapes (e.g., traffic signs) in security footage.

---

## Conclusion
This project provides an efficient and versatile alternative to machine learning-based object detection systems, specifically designed for low-resource environments. By leveraging classical computer vision techniques, dynamic profiles, and adjustable parameters, the solution ensures adaptability and scalability for diverse applications.
