# 2024/nov/03 2.5m

### **Scenario Description: Automated Fruit Detection and Analysis in Agricultural Imaging**

"""**Objective:**  
Develop an automated system for detecting, counting, and analyzing round objects (such as fruits) in agricultural settings using advanced image processing and computer vision techniques. This system aims to enhance yield estimation, monitor crop health, and optimize harvesting processes by leveraging high-resolution images captured from various sources like drones or ground-based cameras.

**Workflow:**

1. **Image Acquisition:**
   - **Data Collection:** Utilize drones, satellite imagery, or fixed cameras to capture high-resolution images of orchards, plantations, or fruit-bearing areas. Ensure consistent lighting conditions and optimal angles to facilitate accurate detection.
   - **Image Input Options:** Support both direct image file paths (`path_image`) and Base64-encoded image strings (`image64`) to provide flexibility in data sourcing.

2. **Preprocessing:**
   - **Resizing:** Use `resize_image` to standardize image dimensions, ensuring uniformity across different datasets and optimizing processing speed without losing critical details.
   - **Aspect Ratio Adjustment:** Apply `crop_image_to_aspect_ratio` to maintain desired aspect ratios (e.g., 4:3 or 3:4) based on image orientation, enhancing the focus on relevant regions.

3. **Image Enhancement:**
   - **Contrast Enhancement:** Implement `apply_clahe` to improve image contrast adaptively, making fruits more distinguishable from the background.
   - **Color Smoothing:** Utilize `smooth_color` to reduce color noise and enhance the clarity of fruit regions, facilitating better segmentation.

4. **Color Filtering and Mask Generation:**
   - **Color Space Conversion:** Convert images to appropriate color spaces (HSV or LAB) using OpenCV's `cvtColor` for effective color-based segmentation.
   - **Color Range Selection:** Employ `filter_color` to isolate target fruits based on predefined color ranges, generating binary masks that highlight potential fruit regions.
   - **Morphological Operations:** Apply functions like `remove_salt_and_pepper`, `close_mask_circle`, and `expand_mask_circle` to refine masks by eliminating noise, closing gaps, and expanding regions to cover entire fruits.

5. **Foreground and Background Segmentation:**
   - **Foreground Masking:** Combine multiple masks using weighted summation to create a comprehensive foreground mask that accurately represents fruit regions.
   - **Background Masking:** Invert the foreground mask to delineate background areas, enabling separate processing if needed.

6. **Object Detection:**
   - **Circle Detection:** Utilize `detect_circles` with OpenCV's `SimpleBlobDetector` to identify circular shapes corresponding to fruits. Parameters like circularity, convexity, and inertia ratio are fine-tuned to enhance detection accuracy.
   - **Overlap Handling:** Implement `remove_overlapping_circles` to eliminate duplicate detections and ensure each fruit is counted only once.
   - **Visualization:** Use `draw_circles` to annotate detected fruits on the original image, providing visual confirmation and aiding in result verification.

7. **K-Means Clustering (Optional):**
   - **Color Quantization:** Apply `kmeans_recolor` to reduce the color palette of the image, simplifying color variations and potentially improving detection performance in complex backgrounds.
   - **Visualization:** Generate clustered images to analyze color distributions and adjust clustering parameters as needed.

8. **EXIF Metadata Extraction:**
   - **Metadata Retrieval:** Use `extract_metadata_EXIF` to obtain valuable information such as GPS coordinates, capture date, camera model, and manufacturer from image metadata.
   - **Data Integration:** Incorporate EXIF data into the final report for contextual insights, enabling spatial and temporal analysis of fruit distribution and growth patterns.

9. **Post-processing and Reporting:**
   - **Result Compilation:** Aggregate detection results, including the number of fruits detected and accuracy metrics, into a structured JSON format using `format_result`.
   - **Image Saving:** Save annotated images with detection overlays and preprocessed images for record-keeping and further analysis using `save_image`.
   - **Interactive Tuning (Debugging):** Provide functions like `tunning_blur`, `tunning_color`, and `tunning_BlobCircles` with OpenCV trackbars for real-time parameter adjustments, facilitating optimization during development and deployment.

10. **Batch Processing and Automation:**
    - **Batch Mode Support:** Enable processing of multiple images in a directory through the `test_directory` function, automating the detection pipeline and generating summarized accuracy reports.
    - **Error Handling:** Implement robust error handling to manage issues like invalid image paths, decoding errors, and missing EXIF data, ensuring the system's resilience and reliability.

**Benefits:**

- **Efficiency:** Automates the time-consuming process of fruit counting and analysis, allowing farmers and agricultural professionals to focus on decision-making and field operations.
- **Accuracy:** Enhances detection precision through advanced filtering, morphological operations, and customizable parameters, reducing human error and ensuring reliable yield estimates.
- **Scalability:** Capable of handling large datasets from extensive agricultural fields, supporting scalability as operations grow.
- **Flexibility:** Accommodates various image input formats and sources, providing versatility in data acquisition and integration.
- **Insightful Reporting:** Integrates EXIF metadata and detailed JSON reports, offering comprehensive insights into fruit distribution, environmental conditions, and system performance.
- **User-Friendly Tuning:** Interactive tuning functions enable users to adjust parameters on-the-fly, tailoring the system to specific agricultural contexts and improving adaptability to different fruit types and environments.
- **Cost-Effective:** Reduces the need for manual labor in fruit counting and monitoring, lowering operational costs and increasing overall productivity.

By implementing this comprehensive system, agricultural stakeholders can achieve enhanced monitoring of crop yields, better resource allocation, and informed strategies for crop management, ultimately contributing to increased agricultural efficiency and sustainability.
"""

import os
import random
from PIL import Image as PILImage
import io, base64, json
import numpy as np

BATCH_MODE = True
PRINT_REPORT_ON_IMAGE = True

VERSION='2024/dec/03 2.5m'

from utils_cv import *
from config_profiles import *


# Configure debugging levels and wait times based on batch mode and debug status
if BATCH_MODE:
    DEBUG_LEVEL = 0
    WAIT_TIME = 1
    TUNNING_BLUR = TUNNING_TEXTURE = TUNNING_FOREGROUND = TUNNING_OBJECT = TUNNING_CIRCLES = False
else:
    TUNNING_BLUR = True
    TUNNING_TEXTURE = True
    TUNNING_FOREGROUND = True
    TUNNING_OBJECT = True
    TUNNING_CIRCLES = True
    DEBUG_LEVEL = 3
    WAIT_TIME = 0

#####################################################################################################
# Main Function
#####################################################################################################



def count_round_objects(path_image='', image64='', profile='ORANGE ', json_exif_str='', json_gps_str=''):
    """
    Processes an image to identify and outline round objects (e.g., fruits) through a series of image processing steps.

    **Scenario:**
    This function is particularly useful in agricultural settings where automated fruit counting and analysis are required.
    For example, farmers can use this function to estimate the yield of apple trees by analyzing images captured by drones.

    **Workflow Logic:**
    1. **Image Loading:**  
       - Accepts either a file path or a Base64-encoded string to load the image.
       - Converts the image to a NumPy array for processing.

    2. **EXIF Metadata Extraction:**  
       - Retrieves metadata such as GPS coordinates, capture date, device model, and manufacturer.
       - Incorporates this information into the result for contextual analysis.

    3. **Configuration Loading:**  
       - Loads color profiles and processing configurations based on the specified profile.
       - Ensures that the processing parameters are tailored to the specific color characteristics of the target objects.

    4. **Image Resizing and Aspect Ratio Adjustment:**  
       - Crops the image to maintain a desired aspect ratio, enhancing the focus on relevant regions.
       - Resizes the image to a maximum resolution to optimize processing speed without compromising essential details.

    5. **Color Smoothing:**  
       - Applies color smoothing to reduce noise and enhance the clarity of target objects.

    6. **Texture Removal:**  
       - Utilizes morphological operations to detect and remove textures that may interfere with object detection.
       - Processes multiple texture layers to ensure clean segmentation.

    7. **Color Amplification:**  
       - Enhances the saturation of specific hues to make target objects more distinguishable from the background.

    8. **Blurring and Contrast Adjustment:**  
       - Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) for contrast enhancement.
       - Removes salt-and-pepper noise and applies Gaussian blur to further smooth the image.

    9. **Foreground Selection:**  
       - Combines multiple color filters to create a comprehensive mask highlighting the foreground objects.
       - Differentiates between certain and probable object regions for more accurate detection.

    10. **Quantization (Optional):**  
        - Applies K-Means clustering to reduce the color palette, simplifying color variations and potentially improving detection accuracy.

    11. **Circle Detection:**  
        - Detects circular shapes corresponding to target objects using blob detection techniques.
        - Handles overlapping detections to ensure each object is counted only once.

    12. **Result Compilation and Saving:**  
        - Draws detected circles on the original image for visualization.
        - Calculates accuracy metrics if ground truth data is available.
        - Saves the processed images and compiles the results into a JSON format for reporting.

    **Arguments:**
        path_image (str): Directory and filename of the image file (optional).
        image64 (str): Base64-encoded string of the image (optional).
        profile (str): Desired color profile for fruit detection (default is 'ORANGE ').
        json_exif_str (str): EXIF data of the image as a JSON-formatted string (optional).
        json_gps_str (str): GPS data of the image as a JSON-formatted string (optional).

    **Returns:**
        str: JSON-formatted string containing image dimensions, metadata, detection results, and other relevant information.
             If an error occurs, returns a JSON string with an error message.
    """
    
    print(f'count_round_objects. version={VERSION}')

    # Ensure that only one of path_image or image64 is provided
    if (path_image == '' and image64 == '') or (path_image != '' and image64 != ''):
        print('Parameters path_image and image64 are mutually exclusive and mandatory.')
        return '{"error": "error"}'

    print(f'  json_exif_str={json_exif_str}')
    print(f'  json_gps_str={json_gps_str}')

    # Load image from Base64 string (suitable for JavaScript, HTML, Ionic)
    print('Trying to load image')
    if image64 != '':
        try:
            img_IO = PILImage.open(io.BytesIO(base64.b64decode(image64)))
        except:
            error = '{"error": "Error reading image in base64"}'
            print(error)
            return error
    # Load image from file path (suitable for Python)
    else:
        try:
            img_IO = PILImage.open(path_image)
        except:
            error = '{"error": "Error reading file ' + path_image + '"}'
            print(error)
            return error

    img_original = np.array(img_IO)
    result_json = { 'dimensions': f'{img_original.shape[0]}x{img_original.shape[1]}px'}

    # Read EXIF data (GPS, device...)
    result_exif = {}
    if extract_metadata_EXIF(img_IO, result_exif):
        print(f"    EXIF: coordinates={result_exif.get('coordinates')}, date={result_exif.get('capture_date')}, model={result_exif.get('mobile_model')}, manufacturer={result_exif.get('mobile_manufacturer')}")
        result_json.update(result_exif)
    img_IO.close()

    file = path_image.split('/')[-1]
    directory = path_image.replace(file, '')

    #####################################################################################################
    print(f'0. Loading configuration:')
    #####################################################################################################
    # Initialize the static configuration variable on the first call
    if not hasattr(count_round_objects, 'cfg'):
        print(f'    Loading profile for first time: {profile}')
        count_round_objects.cfg = load_config(profile)
        
    cfg = count_round_objects.cfg
    if cfg.profile != profile:
        print(f'    Loading new profile: {profile}')
        cfg = load_config(profile)

    #####################################################################################################
    print(f'1. Resize:')
    #####################################################################################################

    images = [img_original]
    masks = []

    # Crop image to maintain desired aspect ratio
    if cfg.aspect_ratio == 0:
        print('    Skipping cropping image to aspect ratio.')
    else:
        print(f'    Cropping image to aspect ratio {cfg.aspect_ratio[0]}x{cfg.aspect_ratio[1]}:')
        img_crop = crop_image_to_aspect_ratio(images[-1], cfg.aspect_ratio)
        if img_original.shape != img_crop.shape:
            print(f'    Original Size: {images[-1].shape}    New Size: {img_crop.shape}')
        images.append(img_crop)

    # Resize image to maximum resolution
    if cfg.max_resolution == 0:
        print('    Skipping Resize')
    else:
        images.append(resize_image(images[-1], cfg.max_resolution))
        print(f'    Original Size: {images[-2].shape}    New Size: {images[-1].shape}')

        if DEBUG_LEVEL >= 5:
            show_mosaic([images[-2], images[-1]], headers=['Original', 'Resized'], window_name='Original vs Resized')

    img_reduced = images[-1]

    #####################################################################################################
    print(f'1.1. Smooth color:')
    #####################################################################################################

    img_before = images[-1].copy()
    img_tmp = img_before

    # Apply color smoothing if configured
    if cfg.smooth_colors == 0:
        print('    Skipping Smooth Color')
    else:
        img_tmp = smooth_color(img_tmp, kernel_size=cfg.smooth_colors, min_brightness=20, max_brightness=210)
    
        if DEBUG_LEVEL >= 2:
            show_mosaic([img_before, img_tmp], 
                        mosaic_dims=(2,1),
                        headers=['Before'],
                        footers=['Smoothed'],
                        window_name=f'Smooth color',
                        max_resolution=800)
            cv2.waitKey(WAIT_TIME)
            
    img_smooth_color = img_tmp
    images.append(img_tmp)

    #####################################################################################################
    print(f'   Remove Texture 1:')
    #####################################################################################################

    # Remove texture using morphological operations if configured
    if cfg.texture_1_kernel_size <= 1:
        print('   Skipping Texture 1')
    else:
        img_tmp = images[-1].copy()
        img_after = np.zeros_like(images[-1])
        mask_tmp = np.zeros_like(img_tmp)

        if TUNNING_TEXTURE:
            img_after, mask_tmp, cfg.texture_1_kernel_size, cfg.texture_1_threshold_value, \
            cfg.texture_1_noise, cfg.texture_1_expand, cfg.texture_1_it = \
            tunning_texture(img_tmp,
                            kernel_size=cfg.texture_1_kernel_size, 
                            threshold_value=cfg.texture_1_threshold_value,
                            noise=cfg.texture_1_noise,
                            expand=cfg.texture_1_expand,
                            it=cfg.texture_1_it)
            
        img_after, mask_tmp = detect_smooth_areas_rgb(img_tmp, 
                                        kernel_size=cfg.texture_1_kernel_size, 
                                        threshold_value=cfg.texture_1_threshold_value, 
                                        noise=cfg.texture_1_noise, 
                                        expand=cfg.texture_1_expand, 
                                        it=cfg.texture_1_it)
        
        images.append(img_after)
        masks.append(~mask_tmp)

    #####################################################################################################
    print(f'1.2. Amplify Color:')
    #####################################################################################################
    # Amplify saturation around a specific hue to enhance target objects
    if cfg.color_amplify_range == 0:
        print('    Skipping Amplify Color')
        img_amplified = images[-1].copy()
    else:
        img_tmp = images[-1].copy()
        img_amplified = amplify_saturation_near_hue(img_tmp, cfg.color_amplify_hue, cfg.color_amplify_range, cfg.color_amplify_increase)
        if DEBUG_LEVEL >= 4:
            show_mosaic([img_amplified, img_tmp], 
                        mosaic_dims=(2,1),
                        headers=['Amplified'],
                        footers=['Original'], 
                        window_name='Amplify Saturation',
                        max_resolution=800)

        images.append(img_amplified)  

    #####################################################################################################
    print(f'2. Blur:')
    #####################################################################################################

    img_tmp = images[-1].copy()

    # Interactive tuning of blur parameters if in debug mode
    if TUNNING_BLUR:
        _, cfg.blur_clahe_grid, cfg.blur_clahe_limit, cfg.blur_salt_pepper, cfg.blur_size = \
            tunning_blur(img_tmp, 
                        clahe_grid=cfg.blur_clahe_grid, 
                        clahe_limit=cfg.blur_clahe_limit, 
                        salt_pepper=cfg.blur_salt_pepper, 
                        blur_size=cfg.blur_size)

    # Apply CLAHE, remove noise, and apply Gaussian blur
    img_tmp = images[-1].copy()
    _, img_tmp, _ = apply_clahe(img_tmp, 
                                tileGridSize=(cfg.blur_clahe_grid, cfg.blur_clahe_grid), 
                                clipLimit=cfg.blur_clahe_limit)
    img_tmp = remove_salt_and_pepper(img_tmp, kernel_size=cfg.blur_salt_pepper)
    img_tmp = blur_image(img_tmp, kernel_size=cfg.blur_size)

    images.append(img_tmp.copy())
    img_blur = img_tmp.copy()
    del img_tmp

    print(f'   Remove Texture 2:')

    # Remove additional texture layers if configured
    if cfg.texture_2_kernel_size <= 1:
        print('   Skipping Texture 2')
    else:
        img_tmp = images[-1].copy()
        img_after = np.zeros_like(images[-1])
        mask_tmp = np.zeros_like(img_tmp)

        if TUNNING_TEXTURE:
            img_after, mask_tmp, cfg.texture_2_kernel_size, cfg.texture_2_threshold_value, \
            cfg.texture_2_noise, cfg.texture_2_expand, cfg.texture_2_it = \
            tunning_texture(img_tmp,
                            kernel_size=cfg.texture_2_kernel_size, 
                            threshold_value=cfg.texture_2_threshold_value,
                            noise=cfg.texture_2_noise,
                            expand=cfg.texture_2_expand,
                            it=cfg.texture_2_it)
        
        img_after, mask_tmp = detect_smooth_areas_rgb(img_tmp, 
                                        kernel_size=cfg.texture_2_kernel_size, 
                                        threshold_value=cfg.texture_2_threshold_value, 
                                        noise=cfg.texture_2_noise, 
                                        expand=cfg.texture_2_expand, 
                                        it=cfg.texture_2_it)
        
        images.append(img_after)
        masks.append(~mask_tmp)


    #####################################################################################################
    print(f'3. Select Foreground')
    #####################################################################################################

    img_before = images[-1].copy()
    img_tmp = img_before
    mask_tmp = np.zeros(img_tmp.shape[:2], dtype=np.int8)

    # Iterate through the list of foreground colors to create masks
    for i in range(len(cfg.foreground_list)):
        cfg.foreground_name = cfg.foreground_list[i][0][0]
        cfg.foreground_weight = cfg.foreground_list[i][0][1]
        color_space = 'LAB' if 'LAB' in cfg.foreground_name else 'HSV'

        if cfg.foreground_weight != 0:
            if TUNNING_FOREGROUND:
                selected_area, discarded_area, mask_selected = \
                    tunning_color(img_tmp, 
                                parameters=cfg.foreground_list[i][1],
                                window_name=f'{cfg.foreground_name} ({cfg.foreground_weight})',
                                color_space=color_space)

            selected_area, discarded_area, mask_selected = filter_color(img_tmp, 
                                                            color_ini=cfg.foreground_list[i][1][0:3], 
                                                            color_end=cfg.foreground_list[i][1][3:6],  
                                                            noise=cfg.foreground_list[i][1][6],
                                                            expand=cfg.foreground_list[i][1][7],
                                                            close=cfg.foreground_list[i][1][8],
                                                            iterations=cfg.foreground_list[i][1][9],
                                                            color_space=color_space)
            if DEBUG_LEVEL >= 4:
                    show_mosaic([selected_area, discarded_area, mask_selected], 
                                mosaic_dims=(3,1),
                                headers=['Selected'],
                                footers=['Mask/Discarded'],
                                window_name=f'Foreground Color Selection {cfg.foreground_name}',
                                max_resolution=800)
            
            # Accumulate mask weights based on configuration
            mask_tmp += np.where(mask_selected > 0, cfg.foreground_weight, 0)

            del selected_area, discarded_area, mask_selected
        print('    .')

    # Create a binary foreground mask
    mask_foreground = np.where(mask_tmp >= 1, 255, 0).astype('uint8')
    if '' in cfg.profile:
        mask_foreground = expand_mask_circle(mask_foreground, kernel_size=7, iterations=1)

    # Apply the foreground mask to the smoothed color image
    img_foreground = cv2.bitwise_and(img_smooth_color, img_smooth_color, mask=mask_foreground)
    img_background = cv2.bitwise_and(img_smooth_color, img_smooth_color, mask=~mask_foreground)
    images.append(img_foreground)
    
    if DEBUG_LEVEL >= 3:
        show_mosaic([img_foreground, img_background], 
                    mosaic_dims=(2,1),
                    headers=['Selected'], 
                    footers=['Discarded'], 
                    window_name='Result Foreground',
                    max_resolution=800)
        cv2.waitKey(WAIT_TIME)

    print(f'   Remove Texture 3:')

    # Remove further texture layers if configured
    if cfg.texture_3_kernel_size <= 1:
        print('   Skipping Texture 3')
    else:
        img_tmp = images[-1].copy()
        img_after = np.zeros_like(images[-1])
        mask_tmp = np.zeros_like(img_tmp)

        if TUNNING_TEXTURE:
            img_after, mask_tmp, cfg.texture_3_kernel_size, cfg.texture_3_threshold_value, \
            cfg.texture_3_noise, cfg.texture_3_expand, cfg.texture_3_it = \
            tunning_texture(img_tmp,
                            kernel_size=cfg.texture_3_kernel_size, 
                            threshold_value=cfg.texture_3_threshold_value,
                            noise=cfg.texture_3_noise,
                            expand=cfg.texture_3_expand,
                            it=cfg.texture_3_it)
        
        img_after, mask_tmp = detect_smooth_areas_rgb(img_tmp, 
                                        kernel_size=cfg.texture_3_kernel_size, 
                                        threshold_value=cfg.texture_3_threshold_value, 
                                        noise=cfg.texture_3_noise, 
                                        expand=cfg.texture_3_expand, 
                                        it=cfg.texture_3_it)
        
        # Expand mask to cover borders
        mask_tmp = np.where(cv2.cvtColor(img_after, cv2.COLOR_RGB2GRAY) > 0, 255, 0).astype('uint8')
        images.append(img_after)
        masks.append(mask_tmp)

        # Refine mask by erosion and expansion
        mask_tmp = erode_mask_circle(mask_tmp, kernel_size=cfg.expand_foreground//2, iterations=1)
        mask_tmp = expand_mask_circle(mask_tmp, kernel_size=cfg.expand_foreground, iterations=1)
        img_after = cv2.bitwise_and(img_reduced, img_reduced, mask=mask_tmp)
        images.append(img_after)
        masks.append(mask_tmp)

        if DEBUG_LEVEL >= 3:
            show_mosaic([masks[-2], images[-2], masks[-1], images[-1]], 
                        mosaic_dims=(2,2),
                        window_name='Foreground',
                        max_resolution=800)
            cv2.waitKey(WAIT_TIME)

    print(f'3.1. Quantization:')

    # Apply K-Means color quantization if configured
    if cfg.quantization_n_colors == 0:
        print('    Skipping Quantization')
    else:
        img_tmp = images[-1].copy()

        # Perform color quantization
        recolored_image, clustered_rgb, clustered_labels = kmeans_recolor(img_tmp, n_clusters=10)

        images.append(recolored_image)  

        if DEBUG_LEVEL >= 2:
            show_mosaic([recolored_image, clustered_rgb, img_tmp], 
                        mosaic_dims=(3,1),
                        headers=['Recolored'],
                        footers=['Original'], 
                        window_name='Quantization',
                        max_resolution=800)
            cv2.waitKey(WAIT_TIME)

    img_foreground = images[-1].copy()
                    
    #####################################################################################################
    print(f'4. Select object:')
    #####################################################################################################

    img_before = images[-1].copy()
    img_tmp = img_before
    mask_tmp = np.zeros(img_tmp.shape[:2], dtype=np.int8)
    mask_certainly_object = np.zeros(img_tmp.shape[:2], dtype=np.uint8)

    # Iterate through the list of target colors to create masks for object detection
    for i in range(len(cfg.color_list)):
        cfg.color_name = cfg.color_list[i][0][0]
        cfg.color_weight = cfg.color_list[i][0][1]

        if cfg.color_weight != 0:
            if TUNNING_OBJECT:
                selected_area, discarded_area, mask_selected = \
                    tunning_color(img_tmp, 
                                parameters=cfg.color_list[i][1],
                                window_name=f'{cfg.color_name} ({cfg.color_weight})')

            selected_area, discarded_area, mask_selected = filter_color(img_tmp, 
                                                            color_ini=cfg.color_list[i][1][0:3], 
                                                            color_end=cfg.color_list[i][1][3:6],  
                                                            noise=cfg.color_list[i][1][6],
                                                            expand=cfg.color_list[i][1][7],
                                                            close=cfg.color_list[i][1][8],
                                                            iterations=cfg.color_list[i][1][9])
            
            # Save a mask for areas that are certain to be objects (e.g., fruits)
            if cfg.color_weight >= 2:
                mask_rounded = cv2.bitwise_or(mask_certainly_object, mask_selected)
                if cfg.smooth_mask_certain >= 3:
                    kernel_size = cfg.smooth_mask_certain
                    mask_rounded = erode_mask_circle(mask_rounded, kernel_size=kernel_size, iterations=1)
                    mask_rounded = expand_mask_circle(mask_rounded, kernel_size=kernel_size, iterations=1)
                    mask_rounded = erode_mask_circle(mask_rounded, kernel_size=make_odd(kernel_size/2), iterations=2)
                    mask_rounded = expand_mask_circle(mask_rounded, kernel_size=make_odd(kernel_size/2), iterations=1)
                mask_certainly_object = mask_rounded.astype('uint8')
                if DEBUG_LEVEL >= 3:
                    show_mosaic([mask_selected, mask_certainly_object], 
                                headers=['Selected Now'],
                                footers=['Added to Certainly Object'],
                                mosaic_dims=(2, 1), 
                                window_name='Mask Certainly Object', 
                                max_resolution=800)
                    cv2.waitKey(WAIT_TIME)
            
            if DEBUG_LEVEL >= 4:
                show_mosaic([selected_area, discarded_area, mask_selected], 
                            mosaic_dims=(3,1),
                            headers=['Selected'],
                            footers=['Mask/Discarded'],
                            window_name=f'Color Selection {cfg.color_name}',
                            max_resolution=800)
            
            # Accumulate mask weights based on configuration
            mask_tmp += np.where(mask_selected > 0, cfg.color_weight, 0)

            del selected_area, discarded_area, mask_selected

        print('    .')

    # Normalize the mask to grayscale
    mask_grayscale = cv2.normalize(mask_tmp, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    if DEBUG_LEVEL >= 3:
        show_mosaic([mask_grayscale, img_before], 
                    mosaic_dims=(2,1),
                    headers=['Mask Fruit'], 
                    footers=['Original'], 
                    window_name='Preview Filter color',
                    max_resolution=800)
        

    # Create a binary mask for objects
    mask_objects = np.where(mask_tmp >= 1, 255, 0).astype('uint8')
    img_after = cv2.bitwise_and(img_before, img_before, mask=mask_objects)
    images.append(img_after)
    masks.append(mask_objects)

    if DEBUG_LEVEL >= 3:
        show_mosaic([masks[-2], images[-2], masks[-1], images[-1]], 
                    mosaic_dims=(2,2),
                    window_name='Objects',
                    max_resolution=800)
        cv2.waitKey(WAIT_TIME)
    
    # Separate selected and discarded areas
    img_fruit = cv2.bitwise_and(img_before, img_before, mask=mask_objects)
    img_discarded = cv2.bitwise_and(img_before, img_before, mask=~mask_objects)
    images.append(img_fruit)
    masks.append(mask_objects)
    
    if DEBUG_LEVEL >= 3:
        show_mosaic([img_fruit, img_discarded], 
                    mosaic_dims=(2,1),
                    headers=['Selected'], 
                    footers=['Discarded'], 
                    window_name='Result Filter Color',
                    max_resolution=800)

    print(f'   Remove Texture 4:')

    # Final texture removal layer if configured
    if cfg.texture_4_kernel_size <= 1:
        print('   Skipping Texture 4')
    else:
        img_tmp = images[-1].copy()
        img_after = np.zeros_like(images[-1])
        mask_tmp = np.zeros_like(img_tmp)

        if TUNNING_TEXTURE:
            img_after, mask_tmp, cfg.texture_4_kernel_size, cfg.texture_4_threshold_value, \
            cfg.texture_4_noise, cfg.texture_4_expand, cfg.texture_4_it = \
            tunning_texture(img_tmp,
                            kernel_size=cfg.texture_4_kernel_size, 
                            threshold_value=cfg.texture_4_threshold_value,
                            noise=cfg.texture_4_noise,
                            expand=cfg.texture_4_expand,
                            it=cfg.texture_4_it)
    
        img_after, mask_tmp = detect_smooth_areas_rgb(img_tmp, 
                                        kernel_size=cfg.texture_4_kernel_size, 
                                        threshold_value=cfg.texture_4_threshold_value, 
                                        noise=cfg.texture_4_noise, 
                                        expand=cfg.texture_4_expand, 
                                        it=cfg.texture_4_it)
        
        images.append(img_after)
        masks.append(~mask_tmp)

    #####################################################################################################
    print(f'6. Blur mask:')
    #####################################################################################################

    img_tmp = images[-1].copy()

    # Convert mask to grayscale and invert
    img_tmp = 255 - cv2.cvtColor(img_tmp, cv2.COLOR_RGB2GRAY)
    img_before = img_tmp

    # Apply noise removal and blurring to the mask
    img_tmp = remove_salt_and_pepper(img_tmp, kernel_size=9)
    img_tmp = blur_image(img_tmp, kernel_size=19)
    img_blur = img_tmp.copy()

    # Improve contrast of the mask
    if cfg.factor_contrast != 1:
        img_contrast_adjusted = np.clip(img_tmp**cfg.factor_contrast, 0, 254).astype('uint8')
        img_tmp = np.where(img_tmp < 255, img_contrast_adjusted, img_tmp).astype('uint8')
        
    # Enhance visibility of certain objects
    img_tmp = np.where(mask_certainly_object > 0, img_tmp**0.92, img_tmp).astype('uint8')

    img_contrast = img_tmp.copy()

    if DEBUG_LEVEL >= 3:
        show_mosaic([img_before, img_blur, img_contrast], 
                    window_name='Blur and contrast mask', 
                    headers=['Original'],
                    footers=['Improved contrast'], mosaic_dims=(3, 1), max_resolution=600)
        cv2.waitKey(WAIT_TIME)

    images.append(img_contrast)


    #####################################################################################################
    print(f'6.1. Fill holes:')
    #####################################################################################################

    img_tmp = images[-1].copy()
    img_before = img_tmp

    img_tmp = fill_holes_with_gray(img_tmp, 100);


    if DEBUG_LEVEL >= 3:
        show_mosaic([img_before, img_tmp], 
                    window_name='Fill Holes', 
                    headers=['Original'],
                    footers=['Filled'], mosaic_dims=(1, 2), max_resolution=600)
        

    img_preprocess = img_tmp
    images.append(img_preprocess)
    
    
    #####################################################################################################
    print(f'7. Detect Circles:')
    #####################################################################################################

    # Interactive tuning of circle detection parameters if in debug mode
    if TUNNING_CIRCLES:
        circles, img_circles, mask_circles, cfg.circle_minCircularity, \
        cfg.circle_minConvexity, cfg.circle_minInertiaRatio, \
        cfg.circle_minArea, cfg.circle_maxArea, cfg.min_radius_circle, \
        cfg.tolerance_overlap = \
            tunning_BlobCircles(image=img_preprocess, img_original=img_reduced,
                                minCircularity=cfg.circle_minCircularity, 
                                minConvexity=cfg.circle_minConvexity, 
                                minInertiaRatio=cfg.circle_minInertiaRatio, 
                                minArea=cfg.circle_minArea, 
                                maxArea=cfg.circle_maxArea,
                                min_radius=cfg.min_radius_circle,
                                tolerance_overlap=cfg.tolerance_overlap)
    
    # Detect circles representing objects
    circles, img_circles, mask_circles = \
        detect_circles(img_preprocess, img_original=img_reduced,
                        minCircularity=cfg.circle_minCircularity, 
                        minConvexity=cfg.circle_minConvexity, 
                        minInertiaRatio=cfg.circle_minInertiaRatio, 
                        minArea=cfg.circle_minArea, 
                        maxArea=cfg.circle_maxArea,
                        min_radius=cfg.min_radius_circle,
                        tolerance_overlap=cfg.tolerance_overlap)
    images.append(img_circles.copy())
    num_circles = len(circles)

    #####################################################################################################
    print("8. Final Result")
    #####################################################################################################

    # Draw detected circles on the original smoothed color image
    img_final = draw_circles(img_smooth_color, circles, show_label=True, solid=False)

    # Calculate accuracy if ground truth is available
    num_found_objects = num_circles

    try:
        # Extract expected number of objects from filename, assuming it's enclosed in brackets
        num_expected_objects = int(file[file.find("[") + 1:file.find("]")])
        accuracy = 100 * num_found_objects / num_expected_objects
        filename_result = f"{file}_result_pd=[{num_found_objects}]acc=[{accuracy:.1f}pct].jpg"
        header_image = f"    Found={num_found_objects} objects     Expected={num_expected_objects}    Accuracy={accuracy:.1f}%"
    except:
        num_expected_objects = -1
        accuracy = -1
        filename_result = f"{file}_result.jpg"
        header_image = f"    Found={num_found_objects} objects"

    header_image += f"    Filename={file}" if file != '' else ''
    
    filename_pre_process = f"{file}_pre.jpg"
    print(header_image)
    
    # Prepare footer with EXIF data or GPS coordinates
    if len(result_exif) > 0:
        footer_image = f"{result_exif}"
    else:
        footer_image = ''

        try:
            data = json.loads(json_gps_str)
            footer_image = f"Latitude: {data['latitude']}, Longitude: {data['longitude']}"
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON string: {e}")
        except KeyError as e:
            print(f"Key not found in JSON data: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

    # Build final mosaic image with report if configured
    if PRINT_REPORT_ON_IMAGE:
        final_mosaic = build_mosaic([img_final],
                    headers=[header_image],
                    footers=[footer_image],
                    max_resolution=cfg.max_resolution)
        img_preprocess = build_mosaic([img_preprocess],
                    headers=[header_image],
                    footers=[footer_image],
                    max_resolution=cfg.max_resolution)
    else:
        final_mosaic = img_final

    if DEBUG_LEVEL >= 1:
        print(f"Final Result:")
        show_mosaic([img_preprocess, final_mosaic], window_name='Final Result', mosaic_dims=(2, 1), max_resolution=800)
        cv2.waitKey(WAIT_TIME)

    
    #####################################################################################################
    print(f'9. Saving files')
    #####################################################################################################

    # Save the final result image
    path_filename_result = os.path.join(directory, filename_result)
    path_filename_pre_process = os.path.join(directory, filename_pre_process)
    save_image(final_mosaic, path_filename_result)
    #save_image(img_preprocess, path_filename_pre_process)
    
    # Resize the original image for return
    image_original_reduced = resize_image(img_original, cfg.resolution_returned)
    print(f'    Original Image Reduced: {img_original.shape}    New Size: {image_original_reduced.shape}')


    #####################################################################################################
    print(f'Print configuration')
    #####################################################################################################

    print('\n\n')

    if DEBUG_LEVEL >= 1 and BATCH_MODE is False:
        print(f"        \n        # Profile:", cfg.profile)
        print(f"        # Quality:")
        print(f"        cfg.quantization_n_colors, cfg.max_resolution, cfg.smooth_colors, cfg.factor_contrast = {cfg.quantization_n_colors, cfg.max_resolution, cfg.smooth_colors, cfg.factor_contrast}")
        print(f"        ")
        print(f"        # Blur:")
        print(f"        cfg.blur_clahe_grid, cfg.blur_clahe_limit, cfg.blur_salt_pepper, cfg.blur_size = {cfg.blur_clahe_grid, cfg.blur_clahe_limit, cfg.blur_salt_pepper, cfg.blur_size}")
        print(f"        ")

        print(f"        # Amplify Saturation:")
        print(f"        cfg.color_amplify_hue, cfg.color_amplify_range, cfg.color_amplify_increase = {cfg.color_amplify_hue, cfg.color_amplify_range, cfg.color_amplify_increase}")
        print(f"        ")

        print(f"        # Foreground selection:")
        for i in cfg.foreground_list:
            print(f"        cfg.foreground_list.append({i})")
        print(f"        ")

        print(f"        # Texture Removal:")
        try:
            print(f"        cfg.texture_1_kernel_size, cfg.texture_1_threshold_value, cfg.texture_1_noise, cfg.texture_1_expand, cfg.texture_1_it = {cfg.texture_1_kernel_size, cfg.texture_1_threshold_value, cfg.texture_1_noise, cfg.texture_1_expand, cfg.texture_1_it}")
            print(f"        cfg.texture_2_kernel_size, cfg.texture_2_threshold_value, cfg.texture_2_noise, cfg.texture_2_expand, cfg.texture_2_it = {cfg.texture_2_kernel_size, cfg.texture_2_threshold_value, cfg.texture_2_noise, cfg.texture_2_expand, cfg.texture_2_it}")
            print(f"        cfg.texture_3_kernel_size, cfg.texture_3_threshold_value, cfg.texture_3_noise, cfg.texture_3_expand, cfg.texture_3_it = {cfg.texture_3_kernel_size, cfg.texture_3_threshold_value, cfg.texture_3_noise, cfg.texture_3_expand, cfg.texture_3_it}")
            print(f"        cfg.texture_4_kernel_size, cfg.texture_4_threshold_value, cfg.texture_4_noise, cfg.texture_4_expand, cfg.texture_4_it = {cfg.texture_4_kernel_size, cfg.texture_4_threshold_value, cfg.texture_4_noise, cfg.texture_4_expand, cfg.texture_4_it}")
        except:
            pass
        print(f"        ")

        print(f"        # Object Selection")
        for i in cfg.color_list:
            print(f"        cfg.color_list.append({i})")
        print(f"        cfg.smooth_mask_certain =  {cfg.smooth_mask_certain}")
        print(f"        ")

        print(f"        # Circle Detection:")
        print(f"        cfg.circle_minCircularity, cfg.circle_minConvexity, cfg.circle_minInertiaRatio, = {cfg.circle_minCircularity, cfg.circle_minConvexity, cfg.circle_minInertiaRatio}")
        print(f"        cfg.circle_minArea, cfg.circle_maxArea =  {cfg.circle_minArea, cfg.circle_maxArea}")
        print(f"        cfg.min_radius_circle, cfg.tolerance_overlap =  {cfg.min_radius_circle, cfg.tolerance_overlap}")
        print(f"        \n\n")        

    #####################################################################################################
    print("10. Calculating Result JSON:")
    #####################################################################################################

    # Read EXIF data again (if needed)
    result_exif = {}
    extract_metadata_EXIF(img_IO, result_exif)
    result_json = { 'dimensions': f'{img_original.shape[0]}x{img_original.shape[1]}px'}
    result_json.update(result_exif)
    img_IO.close()

    # Calculate accuracy
    num_found_objects = num_circles
    report = f"Found {num_found_objects} objects"
    result_json['qt_objects'] = num_found_objects
    result_json['qt_objects_small'] = 0
    result_json['qt_objects_normal'] = num_found_objects
    result_json['qt_objects_big'] = 0
    if accuracy > 0:
        result_json['accuracy'] = accuracy
        result_json['expected_objects'] = num_expected_objects
    
    # Format the result into a comprehensive JSON
    result_json = format_result(imagen_cv2=final_mosaic, imagen_cv2_original_reduced=image_original_reduced, result=result_json)

    print('End of count_round_objects.\n')

    return result_json
    




def display_results(result_json):
    """
    Displays the results of the object counting process.

    **Scenario:**
    After processing an image to count objects (e.g., fruits), this function is used to display the results,
    including any errors, JSON data, and preview images encoded in Base64.

    **Workflow Logic:**
    1. **JSON Parsing:**  
       - Parses the input JSON string to extract result data.
    
    2. **Error Handling:**  
       - Checks for any error messages within the JSON and prints them.
    
    3. **Data Display:**  
       - If available, prints the JSON data and displays the resulting images encoded in Base64.
    
    4. **Visualization:**  
       - Decodes Base64 strings to reconstruct and display images using OpenCV.

    **Parameters:**
        result_json (str): JSON string containing the results of the object counting process.

    **Returns:**
        dict: A dictionary containing parsed result data.
    """
    if (result_json != ''):
        result_dict = json.loads(result_json)

        if "error" in result_dict:
            print(result_dict["error"])

        if "json" in result_dict:
            print(result_dict["json"])

        if "image64" in result_dict:
            print('image64 =', result_dict["image64"][:100])
            img_result_cv2 = np.array(PILImage.open(io.BytesIO(base64.b64decode(result_dict["image64"]))))
            img_result_cv2 = cv2.cvtColor(img_result_cv2, cv2.COLOR_BGR2RGB)


        if "image64_original_reduced" in result_dict:
            print('image64_original_reduced =', result_dict["image64_original_reduced"][:100])
            img_result_original_reduced_cv2 = np.array(PILImage.open(io.BytesIO(base64.b64decode(result_dict["image64_original_reduced"]))))
            img_result_original_reduced_cv2 = cv2.cvtColor(img_result_original_reduced_cv2, cv2.COLOR_BGR2RGB)

        if "text" in result_dict:
            print(result_dict["text"])

    else:
        print('Error in result')

    return result_dict


def format_result(imagen_cv2, imagen_cv2_original_reduced, result):
    """
    Formats the results obtained from object counting into a structured JSON string.

    **Scenario:**
    After detecting and counting objects in an image, this function compiles the results,
    including images and metadata, into a JSON format suitable for reporting or further processing.

    **Workflow Logic:**
    1. **Result Text Compilation:**  
       - Constructs a descriptive text based on available data such as filename, dimensions, GPS coordinates, capture date, and device information.
    
    2. **Image Encoding:**  
       - Converts the final processed image and the reduced original image to Base64-encoded strings for easy embedding and transmission.
    
    3. **JSON Structuring:**  
       - Organizes all relevant information, including text reports and encoded images, into a comprehensive JSON structure.

    **Arguments:**
        imagen_cv2: The final processed image in OpenCV (cv2) format.
        imagen_cv2_original_reduced: The resized original image in OpenCV (cv2) format.
        result: A dictionary containing detection results and metadata.

    **Returns:**
        str: A JSON-formatted string containing the formatted results, text report, and Base64-encoded images.
    """

    if 'original_filename' in result:
        result_text = f"Image '{result['original_filename']}' with dimensions: {result['dimensiones']}:"
    else:
        result_text = f"Image with dimensions: {result['dimensions']}:"

    if 'coordinates' in result:
        result_text += f"\nGPS Coordinates: lat={result['coordinates'][0]}, long={result['coordinates'][1]}"
    else:
        result_text += f"\nGPS Coordinates: Unknown"

    if 'capture_date' in result:
        result_text += f"\nPhoto taken on: {result['capture_date']}"
    else:
        result_text += f"\nPhoto taken on: Unknown date"

    if 'mobile_model' in result and 'mobile_manufacturer' in result:
        result_text += f"\nMobile Device: {result['mobile_model']} / {result['mobile_manufacturer']}."
    else:
        result_text += f"\nMobile Device: Unknown."

    if 'result_file' in result:
        result_text += f"\nProcessing saved in {result['result_file']}."

    result_text += f"\nFound {result['qt_objects']} objects."
    if 'accuracy' in result:
        result_text += f"\nAccuracy: {result['accuracy']}"
        result_text += f"\nGround truth: {result['expected_objects']}"

    # Convert the result dictionary to a JSON string
    result_json = json.dumps(result, indent=2)

    # Encode images to Base64 strings
    image64 = image_to_base64(imagen_cv2)
    image64_original_reduced = image_to_base64(imagen_cv2_original_reduced)

    # Compile all information into a dictionary
    result_dict = {
      'json': result_json,
      'text': result_text,
      'image64': image64,
      'image64_original_reduced': image64_original_reduced
      }

    # Convert the dictionary to a JSON-formatted string
    result_json = json.dumps(result_dict, indent=2)

    return result_json


def test_directory(directory, profile, specific_files=[], limit_files=0):
    """
    Tests the object counting process on a directory of images, optionally filtering specific files or limiting the number of files processed.

    **Scenario:**
    This function is useful for batch processing multiple images in a directory, such as evaluating the system's performance across different datasets or profiles.
    For example, an agricultural researcher can use this function to process all orchard images in a folder and obtain summary accuracy reports.

    **Workflow Logic:**
    1. **File Selection:**  
        - Lists all files in the specified directory.
        - Shuffles the file list randomly.
        - Filters for specific files if provided.
        - Limits the number of files processed if a limit is set.
    
    2. **Processing Loop:**  
        - Iterates through each selected file.
        - Deletes any existing analysis files to avoid duplication.
        - Processes each image using `count_round_objects`.
        - Collects accuracy metrics if available.

    3. **Summary Reporting:**  
        - Calculates the average error across all processed images.
        - Prints a summary report indicating the performance of the specified profile.

    **Arguments:**
        directory (str): Path to the directory containing images to be processed.
        profile (str): Color profile to be used for detection.
        specific_files (list, optional): List of specific filenames to process. Default is empty, meaning all files are processed.
        limit_files (int, optional): Maximum number of files to process. Default is 0, meaning no limit.

    **Returns:**
        str: A summary string indicating the average error and the number of images processed.
    """
    from tqdm import tqdm
    import time    

    # List all files in the directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) ]

    # Shuffle files randomly
    random.shuffle(files)
    if len(specific_files) > 0:
        files = specific_files
    
    if limit_files > 0:
        files = files[:limit_files]
    
    with tqdm(total=len(files)) as pbar:
        accuracies = []

        # Process each image
        for file in files:
            
            print(f'>>>>> Image: {file}')
            path_image = os.path.join(directory, file)

            print(f'Looking for previous analysis:')
            # Skip files that are result or pre-processed images
            if '_result' in file or '_pre.' in file:
                print(f'Deleting previous analysis: {file}')

                try:
                    # Construct the full file path
                    file_path = os.path.join(directory, file)
                    # Delete the file
                    os.remove(file_path)
                    print(f"Deleted previous analysis.")
                except Exception as e:
                    print(f"Error Deleting previous analysis {file}: {e}")
                pbar.total -= 1
            else:                
                print(f'Processing image: {file}')
                pbar.update(1)

                # Count round objects in the image
                result_json = count_round_objects(path_image=path_image, profile=profile)
                result_dict = json.loads(result_json)
                if DEBUG_LEVEL >= 1:
                    display_results(result_json)

                # Collect accuracy metrics if available
                if 'json' in result_dict:
                    result_text_json = json.loads(result_dict['json'])
                    if 'accuracy' in result_text_json:
                        accuracies.append(result_text_json['accuracy'])

        # Compile a summary of accuracies
        accuracy_sumary = f'directory={directory}'
        accuracies_calc = np.array(accuracies)
        if len(accuracies_calc) == 0:
            accuracy_sumary += f'\nNo images found. Average Accuracy of {profile}: 0.0% out of 0 images'
        else:
            accuracy_sumary += f'\nAverage Error of {profile}: {np.mean(np.abs(accuracies_calc-100)):.1f}% out of {len(accuracies_calc)} images'
        print(accuracy_sumary)

        pbar.close()

    return accuracy_sumary
