# Utils: functions created by the author

import numpy as np 
import cv2
import math
import datetime
import PIL.ExifTags as EXIF
from PIL import Image as PILImage
from scipy import ndimage as ndi
import base64

global NOTEBOOK_MODE
global DEBUG_LEVEL

NOTEBOOK_MODE = False
DEBUG_LEVEL = 1

def set_debug_level(level):
    DEBUG_LEVEL = level
    
def set_notebook_mode():
    import matplotlib.pyplot as plt
    import ipywidgets as widgets
    from IPython.display import display
    NOTEBOOK_MODE = True
    

def debug_print(*args, **kwargs):
    """
    Custom debug_print function that only outputs messages if DEBUG_LEVEL > 0.

    Args:
        *args: Positional arguments passed to the debug_print function.
        **kwargs: Keyword arguments passed to the debug_print function.
    """
    if DEBUG_LEVEL > 1:
        print(*args, **kwargs)

def resize_image(image, max_resolution=1200, downscale_factor=0, interpolation=cv2.INTER_AREA):
    """
    Resizes an image while maintaining the aspect ratio.

    **Scenario:**
    Imagine you are developing a mobile application that allows users to upload photos of fruits for analysis. High-resolution images can be large in size, leading to increased loading times and higher bandwidth usage. This function can be used to resize user-uploaded images to a manageable size without distorting them, ensuring faster processing and a better user experience.

    **Logic and Computer Vision Methods:**
    The function first determines the original dimensions of the image. It then calculates a scaling factor based on either a specified downscale factor or a maximum resolution for the longest side of the image. By maintaining the aspect ratio, it ensures that the resized image does not appear stretched or squished. The resizing is performed using OpenCV's `resize` function with the specified interpolation method, which affects the quality and speed of the resizing operation.

    Args:
        image (numpy.ndarray): The image to be resized.
        max_resolution (int): The maximum size for the longest dimension of the image.
        downscale_factor (float): Factor by which the image dimensions are divided.
        interpolation (int): Interpolation method used for resizing.

    Returns:
        numpy.ndarray: The resized image.
        float: The scaling factor applied to the image dimensions.
    """
    # Calculate the image's dimensions.
    height, width = image.shape[:2]

    # Determine the scaling factor.
    if downscale_factor > 0:
        scaling_factor = 1 / downscale_factor
    elif max_resolution > 0:
        scaling_factor = max_resolution / max(height, width)
    else:
        scaling_factor = 1

    # Calculate the new dimensions.
    new_height, new_width = int(height * scaling_factor), int(width * scaling_factor)

    # Perform the resizing operation.
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=interpolation)

    return resized_image

def image_to_base64(image_cv2):
    """
    Encodes an OpenCV image to a Base64 string.

    **Scenario:**
    Consider a web service that processes images of fruits and returns analysis results. To transmit images over the web, especially in JSON format, it's efficient to encode them as Base64 strings. This function converts an OpenCV image into a Base64-encoded string, facilitating easy transmission and storage.

    **Logic and Computer Vision Methods:**
    The function first converts the image from BGR (OpenCV's default color format) to RGB. It then encodes the image into JPEG format using OpenCV's `imencode` function. The resulting byte buffer is then converted to a Base64 string, which can be easily embedded in JSON or transmitted over text-based protocols.

    Args:
        image_cv2 (numpy.ndarray): The image to be encoded.

    Returns:
        str: The Base64-encoded string of the image.
    """
    # Encode the image as JPEG Base64
    # You can also use '.png' for PNG or change the format as needed
    img_cv2_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB) # OpenCV uses BGR internally
        
    _, buffer = cv2.imencode('.jpg', img_cv2_rgb)

    # Convert the image bytes to Base64
    img_base64 = base64.b64encode(buffer)

    # Decode to a UTF-8 string for easier handling or inclusion in JSON
    img_base64_str = img_base64.decode('utf-8')

    return img_base64_str

def build_mosaic(images, headers=[""], footers=[""], mosaic_dims=(0,0), border_size=3, max_resolution=1200):
    """
    Builds a mosaic from a list of OpenCV images with specified dimensions, handling both grayscale
    and color images, and optionally displays headers above each column.
    The intended use is to visualize the results of image *faster* than matplotlib.

    **Scenario:**
    In a fruit detection system, you might process multiple images to detect different types of fruits. To quickly review the results, you can create a mosaic that displays all processed images in a grid format with headers indicating each fruit type. This allows for rapid visual inspection and comparison of detection results.

    **Logic and Computer Vision Methods:**
    The function first ensures that all input images are in color format. It then resizes each image to fit within the specified maximum resolution and adds borders for visual separation. The images are arranged into a grid based on the specified mosaic dimensions. Headers and footers can be added to each column for labeling purposes. The final mosaic is created as a single image by placing each processed image in its designated position within the grid.

    Args:
        images (list or numpy.ndarray): Image, list of images, or NumPy array format.
        headers (list of str): Optional list of strings representing the headers for each column.
        footers (list of str): Optional list of strings representing the footers for each column.
        mosaic_dims (tuple): Tuple of integers (rows, columns) specifying the mosaic's dimensions.
        border_size (int): Integer specifying the border size around each image.
        max_resolution (int): Maximum resolution for resizing images.

    Returns:
        numpy.ndarray: The mosaic image with optional headers and footers.
    """
    if not isinstance(images, list):
        images = [images.copy()]

    assert len(images) > 0, "No images provided for the mosaic."
    assert isinstance(images, (list, np.ndarray)), "Images must be a list or NumPy np.array (single image)."
    assert isinstance(headers, list), "headers must be provided as a list."
    assert isinstance(footers, list), "footers must be provided as a list."
    assert isinstance(mosaic_dims, tuple), "mosaic_dims must be provided as a tuple."
    assert isinstance(headers[0], str), "headers must be strings."
    assert isinstance(footers[0], str), "footers must be strings."

    num_images = len(images)
    m, n = mosaic_dims if mosaic_dims != (0, 0) else (1, num_images)  # Unpack mosaic dimensions (rows, columns).
    header_height = 30 if headers[0] != "" else 0  # Height of the area reserved for headers
    footer_height = 30 if footers[0] != "" else 0  # Height of the area reserved for footers

    reduced_images = []
    for img in images:
        # Supports passing RGB image as parameter, not only [image]
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = cv2.cvtColor(img[...,0], cv2.COLOR_GRAY2RGB)

        resized_img = resize_image(img, max_resolution=max_resolution//n)
        bordered_img = cv2.copyMakeBorder(resized_img, border_size, border_size, 
                                          border_size, border_size, 
                                          cv2.BORDER_CONSTANT, 
                                          value=[100, 100, 100])
        reduced_images.append(bordered_img)

    img_height, img_width = reduced_images[0].shape[:2]
    mosaic_height = img_height * m + header_height + footer_height
    mosaic_width = img_width * n
    mosaic = np.zeros((mosaic_height, mosaic_width, 3), dtype=np.uint8)

    if len(reduced_images) > m * n:
        selected_indices = np.random.choice(len(reduced_images), size=m * n, replace=False)
    else:
        selected_indices = range(len(reduced_images))

    for idx, i in enumerate(selected_indices):
        img = reduced_images[i]
        row = idx // n
        col = idx % n
        start_y = row * img_height + header_height  # Adjust start position for the header height
        mosaic[start_y:start_y + img_height, col * img_width:(col + 1) * img_width] = img

    # Draw headers if provided
    if len(headers) > 0:
        for idx, header in enumerate(headers[:n]):  # Ensure we don't exceed the number of columns
            if len(header) > 0:
                cv2.putText(mosaic, header, (idx * img_width + 10, header_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), thickness=2)

    # Draw footers if provided
    if len(footers) > 0:
        for idx, footer in enumerate(footers[:n]):  # Ensure we don't exceed the number of columns
            if len(footer) > 0:
                cv2.putText(mosaic, footer, (idx * img_width + 10, mosaic_height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), thickness=2)

    return mosaic

def make_odd(number):
    """
    Adjusts the given number to ensure it is odd.

    **Scenario:**
    In image processing, certain operations like filtering or morphological transformations require odd-sized kernels to have a central pixel. This function ensures that any given number is odd, which is essential for maintaining symmetry in these operations.

    **Logic and Computer Vision Methods:**
    The function checks if the provided number is even. If it is, it increments the number by one to make it odd. This simple adjustment ensures compatibility with algorithms that necessitate odd-sized parameters for proper centering.

    Args:
        number (int or float): The input number.

    Returns:
        int: An odd number. If the input number is odd, it is returned as is. If the input number is even, 1 is added to make it odd.
    """
    number = int(number)
    number = int(number + 1 if number % 2 == 0 else number)
    return number

def remove_salt_and_pepper(mask, kernel_size=2):
    """
    Applies a median filter to the mask to remove "salt and pepper" noise.

    **Scenario:**
    When processing binary masks for fruit detection, random noise pixels (salt and pepper noise) can lead to false detections or fragmented regions. This function cleans the mask by removing such noise, resulting in smoother and more accurate mask regions.

    **Logic and Computer Vision Methods:**
    The function first ensures that the kernel size is a positive odd number, which is required for the median filter to have a central pixel. It then applies OpenCV's `medianBlur` function, which replaces each pixel's value with the median value of the neighboring pixels defined by the kernel. This effectively removes isolated noise pixels while preserving the edges of the mask regions.

    Args:
        mask (numpy.ndarray): The input binary image mask to be denoised.
        kernel_size (int): The size of the kernel. Must be a positive odd number.

    Returns:
        numpy.ndarray: The denoised image mask after applying the median filter.
    """
    assert kernel_size > 0, "Kernel size must be a positive odd number"
    kernel_size = make_odd(kernel_size)
    # Kernel size must be a positive odd number
    return cv2.medianBlur(mask, kernel_size)

def expand_mask_circle(mask, kernel_size=20, iterations=1):
    """
    Dilates the mask to increase the size of the white regions and decrease the size of black holes.

    **Scenario:**
    In fruit detection, after initial masking, some fruit regions might be slightly underrepresented due to shadows or lighting variations. Dilating the mask can help in expanding these regions to ensure complete coverage of the fruits, making subsequent detection steps more reliable.

    **Logic and Computer Vision Methods:**
    The function checks if the kernel size is greater than zero. If so, it ensures the kernel size is odd and creates an elliptical structuring element. It then applies OpenCV's `dilate` function, which expands the white regions in the mask based on the structuring element. Multiple iterations can be applied to achieve the desired level of expansion.

    Args:
        mask (numpy.ndarray): The input binary image mask to be dilated.
        kernel_size (int): The size of the kernel, determining the extent of dilation.
        iterations (int): The number of times dilation is applied.

    Returns:
        numpy.ndarray: The dilated image mask.
    """
    if kernel_size > 0:
        kernel_size = make_odd(kernel_size)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask_transformed = cv2.dilate(mask, kernel, iterations=iterations)
    else:
        mask_transformed = mask
    return mask_transformed

def expand_mask_rectangle(mask, kernel_size=(33, 13), iterations=1):
    """
    Dilates the mask using a rectangular structuring element to increase the size of the white regions.

    **Scenario:**
    In scenarios where the objects of interest (e.g., fruits) have elongated shapes or are closely packed, using a rectangular kernel can help in expanding the mask horizontally or vertically, ensuring better coverage and connectivity between adjacent objects.

    **Logic and Computer Vision Methods:**
    The function checks if the kernel size dimensions are greater than zero. It ensures each dimension is odd and creates a rectangular structuring element. OpenCV's `dilate` function is then applied using this kernel, which expands the white regions in the mask in the shape of the rectangle. Multiple iterations can further enhance the expansion effect.

    Args:
        mask (numpy.ndarray): The input binary image mask to be dilated.
        kernel_size (tuple): The size of the kernel in (width, height), determining the extent of dilation.
        iterations (int): The number of times dilation is applied.

    Returns:
        numpy.ndarray: The dilated image mask.
    """
    if kernel_size > (0, 0):
        kernel_size1 = make_odd(kernel_size[0])
        kernel_size2 = make_odd(kernel_size[1])
        kernel_size = (kernel_size1, kernel_size2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        mask_transformed = cv2.dilate(mask, kernel, iterations=iterations)
    else:
        mask_transformed = mask
    return mask_transformed

def erode_mask_circle(mask, kernel_size=21, iterations=1):
    """
    Erodes the mask to decrease the size of the white regions and increase the size of black holes.

    **Scenario:**
    After masking and dilation, there might be unwanted white regions or noise within the mask. Eroding the mask helps in shrinking the white areas, removing small false positives, and refining the boundaries of the detected fruits for more accurate detection.

    **Logic and Computer Vision Methods:**
    Similar to dilation, erosion uses a structuring element to reduce the size of white regions. The function creates an elliptical kernel and applies OpenCV's `erode` function, which systematically shrinks the white regions based on the kernel. Multiple iterations can intensify the erosion effect, effectively removing smaller unwanted areas.

    Args:
        mask (numpy.ndarray): The input binary image mask to be eroded.
        kernel_size (int): The size of the kernel, determining the extent of erosion.
        iterations (int): The number of times erosion is applied.

    Returns:
        numpy.ndarray: The eroded image mask.
    """
    if kernel_size > 0:
        kernel_size = make_odd(kernel_size)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask_transformed = cv2.erode(mask, kernel, iterations=iterations)
    else:
        mask_transformed = mask
    return mask_transformed

def close_mask_circle(mask, kernel_size=20, iterations=1):
    """
    Applies morphological closing to the mask to close small holes and join nearby white regions.

    **Scenario:**
    In fruit detection, there might be small gaps or holes within the masked regions due to uneven lighting or partial occlusions. Closing the mask helps in filling these small holes and connecting adjacent regions, resulting in a more coherent and complete mask of each fruit.

    **Logic and Computer Vision Methods:**
    Morphological closing is a combination of dilation followed by erosion. The function creates an elliptical structuring element and applies OpenCV's `morphologyEx` with the `MORPH_CLOSE` operation. This sequence fills small holes and bridges gaps between white regions, enhancing the overall quality of the mask.

    Args:
        mask (numpy.ndarray): The input binary image mask to be closed.
        kernel_size (int): The size of the kernel, determining the extent of closing.
        iterations (int): The number of times closing is applied.

    Returns:
        numpy.ndarray: The closed image mask.
    """
    if kernel_size > 0:
        kernel_size = make_odd(kernel_size)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask_transformed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    else:
        mask_transformed = mask
    return mask_transformed

def open_mask_circle(mask, kernel_size=20, iterations=1):
    """
    Applies morphological opening to the mask to separate nearby white regions and remove small white spots.

    **Scenario:**
    After initial masking, there might be small unwanted white spots or noise within the mask. Morphological opening helps in removing these small artifacts and separating closely positioned fruits, ensuring each fruit is distinctly represented in the mask.

    **Logic and Computer Vision Methods:**
    Morphological opening is a combination of erosion followed by dilation. The function creates an elliptical structuring element and applies OpenCV's `morphologyEx` with the `MORPH_OPEN` operation. This process removes small white regions and smoothens the boundaries of larger regions without significantly altering their size.

    Args:
        mask (numpy.ndarray): The input binary image mask to be opened.
        kernel_size (int): The size of the kernel, determining the extent of opening.
        iterations (int): The number of times opening is applied.

    Returns:
        numpy.ndarray: The opened image mask.
    """
    if kernel_size > 0:
        kernel_size = make_odd(kernel_size)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask_transformed = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    else:
        mask_transformed = mask
    return mask_transformed

def select_and_invert_mask(img_original, mask):
    """
    Applies a mask to an image and inverts the mask.

    **Scenario:**
    In fruit detection, you might want to isolate the fruit regions from the background. This function inverts the given mask to select only the unmasked (background) areas or the masked (fruit) areas based on your requirements, allowing for versatile image processing tasks such as highlighting fruits or removing backgrounds.

    **Logic and Computer Vision Methods:**
    The function first inverts the mask by setting all non-zero (masked) regions to zero and vice versa, creating a binary mask with opposite meaning. It then uses OpenCV's `bitwise_and` to apply this inverted mask to the original image, effectively isolating the desired regions. The result is the original image with the mask applied and the inverted mask itself for further use.

    Args:
        img_original (numpy.ndarray): The original image to which the mask will be applied.
        mask (numpy.ndarray): A mask image where non-zero values indicate the areas to mask.

    Returns:
        tuple: A tuple containing:
            - img_mask (numpy.ndarray): The image after the mask has been applied.
            - mask_final (numpy.ndarray): The inverted mask used for masking the image.
    """
    
    # Invert the mask: change all non-zero values (masked areas) to 0, and all 0 values to 255.
    # The result is a binary mask with the opposite meaning, where 0 indicates the areas to keep.
    mask_final = np.array(np.where((mask[..., 0] > 0) | 
                                   (mask[..., 1] > 0) | 
                                   (mask[..., 2] > 0),
                                   0, 255), dtype=np.uint8)
    
    # Apply the inverted mask to the original image, keeping only the regions that were not masked.
    img_mask = cv2.bitwise_and(img_original, img_original, mask=mask_final)

    # Return the image with the mask applied, and the inverted mask itself.
    return img_mask, mask_final

def inRange_LAB(image, color_ini, color_end):
    """
    Generates a binary mask where the pixels of an image in the LAB color space 
    that fall within a specified color range are set to 255 (white), 
    and all others are set to 0 (black).

    **Scenario:**
    When detecting ripe fruits like bananas or tomatoes, specific color ranges in the LAB color space can help differentiate them from the background. This function creates a mask that highlights only those pixels within the desired color range, facilitating accurate fruit segmentation.

    **Logic and Computer Vision Methods:**
    The function checks each pixel's LAB values against the specified lower and upper bounds. Using NumPy's `where`, it sets pixels within the range to 255 (white) and others to 0 (black), effectively creating a binary mask. This mask can then be used for further processing, such as morphological operations or object detection.

    Args:
        image (numpy.ndarray): The input image in LAB color space.
        color_ini (tuple): A 3-element tuple representing the starting (minimum) LAB values of the desired color range.
        color_end (tuple): A 3-element tuple representing the ending (maximum) LAB values of the desired color range.

    Returns:
        numpy.ndarray: A binary mask (image) of the same size as the input image, 
                       where pixels within the color range are white and others are black.
    """
    
    # Convert conditions to a binary mask, where pixels within the specified LAB color range
    # are set to 255 (white) and others are set to 0 (black). The intermediate calculations
    # use broadcasting to create a condition np.array, which is then converted into the final mask.
    return np.array(np.where(
        (image[..., 0] >= color_ini[0]) & (image[..., 0] <= color_end[0]) &
        (image[..., 1] >= color_ini[1]) & (image[..., 1] <= color_end[1]) &
        (image[..., 2] >= color_ini[2]) & (image[..., 2] <= color_end[2]),
        255, 0), dtype=np.uint8)

def transform_mask(mask, iterations=1, pepper_kernel_size=3, close_kernel_size=6, expand_kernel_size=9, show=False):
    """
    Applies a sequence of morphological transformations to a binary mask.

    **Scenario:**
    After generating an initial mask for fruit detection, the mask might contain noise or incomplete regions. This function refines the mask by removing noise, closing gaps, and expanding regions to ensure accurate and clean fruit segmentation, which is crucial for reliable detection and counting.

    **Logic and Computer Vision Methods:**
    The function maintains a history of mask transformations for potential visualization. It first removes "salt and pepper" noise using a median filter. Depending on the number of iterations and the sign of the kernel sizes, it applies morphological closing or opening, followed by dilation or erosion. This sequential application of morphological operations helps in refining the mask by eliminating small artifacts and ensuring the mask accurately represents the fruit regions.

    Args:
        mask (numpy.ndarray): The input binary mask to be transformed.
        iterations (int): The number of times to apply the transformations.
        pepper_kernel_size (int): The kernel size for removing salt and pepper noise.
        close_kernel_size (int): The kernel size for morphological closing.
        expand_kernel_size (int): The kernel size for mask dilation (expansion).
        show (bool): If True, display the sequence of mask transformations.

    Returns:
        numpy.ndarray: The transformed mask after all specified operations.
    """
    # Initialize a list to store intermediate masks
    mask_history = [mask.copy()]

    # Apply transformations if iterations are specified
    if iterations != 0:
        # Remove "salt and pepper" noise if kernel size is greater than 0
        if pepper_kernel_size > 0:
            mask_history.append(remove_salt_and_pepper(mask_history[-1], kernel_size=pepper_kernel_size))
        
        # If iterations are negative, set kernel sizes for erosion and opening
        if iterations < 0:
            close_kernel_size = -close_kernel_size if close_kernel_size > 0 else close_kernel_size
            expand_kernel_size = -expand_kernel_size if expand_kernel_size > 0 else expand_kernel_size

        # Perform absolute value of iterations for erosion or dilation
        for i in range(abs(iterations)):
            # Apply morphological closing or opening based on the kernel size sign
            if close_kernel_size > 0:
                mask_history.append(close_mask_circle(mask_history[-1], kernel_size=close_kernel_size))
            elif close_kernel_size < 0:
                mask_history.append(open_mask_circle(mask_history[-1], kernel_size=abs(close_kernel_size)))

            # Apply dilation (expansion) or erosion based on the kernel size sign
            if expand_kernel_size > 0:
                mask_history.append(expand_mask_circle(mask_history[-1], kernel_size=expand_kernel_size))
            elif expand_kernel_size < 0:
                mask_history.append(erode_mask_circle(mask_history[-1], kernel_size=abs(expand_kernel_size)))

        # Optionally display the sequence of transformations
        if show:
            debug_print(f"Detail of transformations on mask:")
            show_mosaic(mask_history, mosaic_dims=(math.ceil(len(mask_history)/2), 2))
        
    # Return the final mask after transformations
    return mask_history[-1]

def filter_color(image, color_ini, color_end, iterations=1, 
                    noise=0, close=0, expand=0, color_space='HSV', show=False):
    """
    Filters an image based on a specified color range in either HSV or LAB color space and applies
    a series of morphological transformations to refine the resulting mask.

    **Scenario:**
    In detecting ripe fruits like apples or oranges, specific color ranges can help isolate them from the background. This function allows filtering the image based on color ranges in HSV or LAB spaces, followed by morphological operations to clean and enhance the mask, ensuring accurate fruit segmentation.

    **Logic and Computer Vision Methods:**
    The function first converts the image to the specified color space (HSV or LAB). It then normalizes the color parameters to match the expected ranges of the color space. It handles inverted color ranges where the lower bound is greater than the upper bound by creating two separate masks and combining them. After generating the initial mask using color thresholds, it applies morphological transformations such as noise removal, closing, and expansion to refine the mask. The final mask is used to extract the selected and discarded areas from the original image.

    Args:
        image (numpy.ndarray): The image to filter, in RGB color space.
        color_ini (tuple): The lower bound of the color range to filter.
        color_end (tuple): The upper bound of the color range to filter.
        iterations (int): Number of times to apply morphological transformations.
        noise (int): The kernel size for salt-and-pepper noise removal; must be positive odd or zero.
        close (int): The kernel size for morphological closing; used to close small holes or connect close objects.
        expand (int): The kernel size for morphological dilation; used to expand white regions.
        color_space (str): The color space ('HSV' or 'LAB') of the image and color range parameters.
        show (bool): If True, intermediate and final results are displayed.

    Returns:
        tuple: A tuple containing the image with the color selected, the image with the color discarded, 
               and the final binary mask used for selection.
    """
    
    # Ensure color space is valid
    assert color_space in ['HSV', 'LAB'], "parameter color_space must be 'HSV' or 'LAB'"
    
    img_RGB = image.copy()
    noise = make_odd(noise)
    if color_space == 'HSV':
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    elif color_space == 'LAB':
        image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # --- Normalize color parameters
    if color_space == 'HSV':
        # Validate range of colors
        str_color_format =  "HSV format: Hue=[0:360]  Saturation=[0:100]  Brightness=[0:100]."
        assert color_ini[0] >= 0   and color_end[0] >= 0,   f"{str_color_format} The H you informed is negative."
        assert color_ini[0] <= 360 and color_end[0] <= 360, f"{str_color_format} The H you informed is greater than 360."
        assert color_ini[1] >= 0   and color_end[1] >= 0,   f"{str_color_format} The S you informed is negative."
        assert color_ini[1] <= 360 and color_end[1] <= 100, f"{str_color_format} The S you informed is greater than 100."
        assert color_ini[2] >= 0   and color_end[2] >= 0,   f"{str_color_format} The V you informed is negative."
        assert color_ini[2] <= 360 and color_end[2] <= 100, f"{str_color_format} The V you informed is greater than 100."

        color_ini = (round(color_ini[0]/2.0), round(color_ini[1]*2.55), round(color_ini[2]*2.55))
        color_end = (round(color_end[0]/2.0), round(color_end[1]*2.55), round(color_end[2]*2.55))
        #debug_print("HSV: Color Parameters Normalized:")
        #debug_print(f"   color_ini={color_ini}     color_end={color_end}")
    elif color_space == 'LAB':
        # Validate range of colors
        str_color_format =  "LAB format: L=[0:100]  A=[-128:127]  B=[-128:127]."
        assert color_ini[0] >= 0    and color_end[0] >= 0,   f"{str_color_format} The L you informed is negative."
        assert color_ini[0] <= 100  and color_end[0] <= 100, f"{str_color_format} The L you informed is greater than 100."
        assert color_ini[1] >= -128 and color_end[1] >= -128,   f"{str_color_format} The A you informed is less than -128."
        assert color_ini[1] <= 127  and color_end[1] <= 127, f"{str_color_format} The A you informed is greater than 127."
        assert color_ini[2] >= -128 and color_end[2] >= -128,   f"{str_color_format} The B you informed is less than -128."
        assert color_ini[2] <= 127  and color_end[2] <= 127, f"{str_color_format} The B you informed is greater than 127."

        image[...,0] = np.array(image[...,0]/2.55, dtype=np.uint8)
        image[...,1] = np.array(image[...,1]-128, dtype=np.int8)
        image[...,2] = np.array(image[...,2]-128, dtype=np.int8)
        image = np.array(image, dtype=np.int8)

        # debug_print("LAB: Image normalized:")
        # debug_print(f"  image[...,0]=[{image[...,0].min()}:{image[...,0].max()}]")
        # debug_print(f"  image[...,1]=[{image[...,1].min()}:{image[...,1].max()}]")
        # debug_print(f"  image[...,2]=[{image[...,2].min()}:{image[...,2].max()}]")
    

    # --- Add support to inverted ranges. Example: Interval 200 to 100 will select 200 to 255 and 0 to 100
    color_ini1, color_end1 = np.array(color_ini), np.array(color_end)
    color_ini2, color_end2 = color_ini1.copy(), color_end1.copy()
    #debug_print(f"color_ini1={color_ini1}   color_end1={color_end1}")
    #debug_print(f"color_ini2={color_ini2}   color_end2={color_end2}")
    detected_inverted_interval = False

    for i in range(len(color_ini)):
        if color_ini1[i] > color_end1[i]:
            detected_inverted_interval = True
            color_ini1[i] = 0
            color_end1[i] = color_end[i]

            color_ini2[i] = color_ini[i]
            color_end2[i] = 255

            #debug_print(f"color_ini1={color_ini1}   color_end1={color_end1}")
            #debug_print(f"color_ini2={color_ini2}   color_end2={color_end2}")

    # --- Apply range to the image getting the mask
    if detected_inverted_interval:
        if color_space == 'HSV':
            mask_a = cv2.inRange(image, color_ini1, color_end1)
            mask_b = cv2.inRange(image, color_ini2, color_end2)
        elif color_space == 'LAB':
            mask_a = inRange_LAB(image, color_ini1, color_end1)
            mask_b = inRange_LAB(image, color_ini2, color_end2)

        mask = cv2.bitwise_or(mask_a, mask_b)
        #show_mosaic([mask_a, mask_b, mask], headers=["Mask A", "Mask B", "Mask A or B"])
    else:
        # Then we apply the threshold to the image
        if color_space == 'HSV':
            mask = cv2.inRange(image, color_ini, color_end)
        elif color_space == 'LAB':
            mask = inRange_LAB(image, color_ini, color_end)

    
    # --- Apply transformations to the mask (erode, dilate, etc.)
    mask = transform_mask(mask, iterations=iterations, pepper_kernel_size=noise, close_kernel_size=close, expand_kernel_size=expand, show=show)
    selected_area = cv2.bitwise_and(img_RGB, img_RGB, mask=mask)
    discarded_area = cv2.bitwise_and(img_RGB, img_RGB, mask=~mask)

    return selected_area, discarded_area, mask

def apply_clahe(img_RGB, tileGridSize=(100, 100), clipLimit=2.0):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to the V channel of an HSV image.

    **Scenario:**
    In images where lighting conditions are uneven, certain areas might be too dark or too bright, obscuring details of the fruits. Applying CLAHE enhances the local contrast of the image, making the features of the fruits more distinguishable and improving the accuracy of subsequent detection algorithms.

    **Logic and Computer Vision Methods:**
    The function converts the RGB image to HSV color space and extracts the V (brightness) channel. CLAHE is then applied to this channel to enhance contrast while preventing over-amplification of noise. The enhanced V channel replaces the original one, and the image is converted back to RGB. This process improves the visibility of details in different regions of the image without affecting the color information.

    Args:
        img_RGB (numpy.ndarray): The input image in RGB color space.
        tileGridSize (tuple): The size of the grid for the tiles used by CLAHE.
        clipLimit (float): The threshold for contrast limiting.

    Returns:
        tuple: A tuple containing:
            - img_GRAY_clahe (numpy.ndarray): The grayscale image after applying CLAHE.
            - img_RGB_clahe (numpy.ndarray): The HSV image converted back to RGB color space after applying CLAHE.
            - img_HSV_clahe (numpy.ndarray): The HSV image with the V channel replaced by the CLAHE-processed channel.
    """
    img_HSV = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2HSV)

    # Extract the V channel (grayscale equivalent) from the HSV image.
    img_GRAY = img_HSV[:, :, 2]

    if clipLimit == 0 or tileGridSize < (3,3):
        debug_print("CLAHE not applied. clipLimit must be greater than 0 and tileGridSize must be at least (3,3).")
        return img_GRAY, img_RGB, img_HSV
    
    # Apply CLAHE to the grayscale image.
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    img_GRAY_clahe = clahe.apply(img_GRAY)

    # Replace the V channel in the original HSV image with the CLAHE-processed channel.
    img_HSV_clahe = img_HSV.copy()
    img_HSV_clahe[:, :, 2] = img_GRAY_clahe

    # Convert the HSV image with the updated V channel back to RGB color space.
    img_RGB_clahe = cv2.cvtColor(img_HSV_clahe, cv2.COLOR_HSV2RGB)

    return img_GRAY_clahe, img_RGB_clahe, img_HSV_clahe

def blur_image(image, mask=None, kernel_size=15, sigmaX=0):
    """
    Smooths an image within the given mask.

    **Scenario:**
    In fruit detection, you might want to blur certain regions of the image to reduce noise or to focus on specific areas. For instance, blurring the background while keeping the fruits sharp can enhance detection accuracy.

    **Logic and Computer Vision Methods:**
    The function applies a Gaussian blur to the entire image using the specified kernel size and standard deviation. If a mask is provided, it blends the blurred image with the original image based on the mask, ensuring that only the regions specified by the mask are smoothed. This selective blurring is achieved using NumPy's `where` function to apply the blurred pixels where the mask is active.

    Args:
        image (numpy.ndarray): The input image to be blurred.
        mask (numpy.ndarray, optional): A binary mask where the region to be smoothed is white and the rest is black.
        kernel_size (int): The size of the Gaussian kernel. Default is 15.
        sigmaX (float): Gaussian kernel standard deviation in X direction. Default is 0.

    Returns:
        numpy.ndarray: The smoothed image.
    """
    kernel_size = (kernel_size, kernel_size)

    # Apply Gaussian blur to the entire image
    smoothed_image = cv2.GaussianBlur(image, kernel_size, sigmaX)

    if mask is None:
        final_image = smoothed_image
    else:
        # Ensure the mask is in binary format [0:1]
        mask = mask // 255

        # Blend the original image and the smoothed image using the mask
        final_image = np.where(mask[:, :, None] == 1, smoothed_image, image)

    return final_image

import cv2
import numpy as np

def amplify_saturation_near_hue(image, target_hue, hue_range, max_increase_pct):
    """
    Amplifies the saturation of pixels within a specific hue range of an image,
    focusing more intensely on a target hue. Considers hue scale looping at 360 degrees.

    **Scenario:**
    To make ripe fruits like oranges stand out more in an image, you might want to increase their color saturation. This function enhances the saturation of colors within a specified hue range, making the target fruits appear more vibrant against the background.

    **Logic and Computer Vision Methods:**
    The function first adjusts the target hue and hue range to match OpenCV's hue scale (0-180). It converts the image to HSV color space and calculates the distance of each pixel's hue from the target hue, accounting for hue wrapping. Pixels within the specified hue range have their saturation and value increased proportionally based on their proximity to the target hue. This selective amplification enhances the vibrancy of the target colors without affecting the entire image.

    Args:
        image (numpy.ndarray): The input image in RGB color space.
        target_hue (int): The target hue value (0-360), which will be adjusted to OpenCV's 0-180 range.
        hue_range (int): The range around the target hue to consider for amplification.
        max_increase_pct (float): The maximum percentage (as a decimal, 0.5 for 50%) to increase saturation by.

    Returns:
        numpy.ndarray: The modified image with amplified saturation in RGB color space.
    """
    # Adjust target hue to OpenCV's hue range
    target_hue = target_hue / 2
    hue_range = hue_range / 2

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Calculate hue distances in a vectorized manner
    hue_distances = np.abs(hsv_image[:, :, 0].astype(np.float32) - target_hue)
    hue_distances = np.minimum(hue_distances, 180 - hue_distances)  # Account for hue looping

    # Identify pixels within the hue range to amplify saturation
    within_range = hue_distances <= (hue_range / 2)

    # Calculate proximity percentages and increase factors
    proximity_pct = 1 - (hue_distances / (hue_range / 2))
    increase_factors = 1 + (max_increase_pct * proximity_pct)

    # Apply saturation amplification only to pixels within the hue range
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * np.where(within_range, increase_factors, 1), 0, 255).astype(np.uint8)
    hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] * np.where(within_range, increase_factors, 1), 0, 255).astype(np.uint8)

    # Convert back to RGB color space
    modified_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

    return modified_image

def detect_circles(image, img_original, minCircularity=0.3, minConvexity=0.5, minInertiaRatio=0.3, minArea=300, maxArea=15000, min_radius=10, max_radius=0, tolerance_overlap=0.1):
    """
    Detects circular blobs in an image using the SimpleBlobDetector algorithm.

    **Scenario:**
    In fruit detection, particularly for round fruits like apples or oranges, detecting circular shapes can help in accurately identifying and counting the fruits present in an image. This function automates the detection of such circular regions, facilitating tasks like inventory management or quality control.

    **Logic and Computer Vision Methods:**
    The function configures OpenCV's `SimpleBlobDetector` with parameters that filter blobs based on circularity, convexity, inertia ratio, and area. It processes the image to enhance circle detection by inverting grayscale values and adding an inner border. Multiple detection attempts with varying thresholds improve the chances of detecting circles of different sizes and qualities. Detected circles are refined by enforcing minimum and maximum radius constraints and removing overlapping circles based on a specified tolerance, ensuring each detected circle corresponds to a distinct fruit.

    Args:
        image (numpy.ndarray): The input image in which to detect circles (RGB or Grayscale/Binary Mask).
        img_original (numpy.ndarray): A copy of the original image for drawing detected circles.
        minCircularity (float): The minimum circularity of detected blobs.
        minConvexity (float): The minimum convexity of detected blobs.
        minInertiaRatio (float): The minimum inertia ratio of detected blobs.
        minArea (int): The minimum area of detected blobs.
        maxArea (int): The maximum area of detected blobs.
        min_radius (int): The minimum radius of detected circles.
        max_radius (int): The maximum radius of detected circles. If 0, it is set to twice the minimum radius.
        tolerance_overlap (float): The overlap tolerance to consider circles as overlapping.

    Returns:
        tuple: A tuple containing:
            - circles_result (list): A list of detected circles, each represented as ((x, y), radius).
            - img_delimited (numpy.ndarray): The original image with detected circles drawn.
            - img_gray_inverted (numpy.ndarray): The processed grayscale image used for detection.
    """
    assert 0.0 <= minCircularity <= 1.0, "minCircularity must be in the range [0.0, 1.0]"
    assert 0.0 <= minConvexity <= 1.0, "minConvexity must be in the range [0.0, 1.0]"
    assert 0.0 <= minInertiaRatio <= 1.0, "minInertiaRatio must be in the range [0.0, 1.0]"

    if minArea > maxArea or minArea == 0:
        return [], image, np.zeros_like(image)
    
    if max_radius == 0:
        max_radius = int(min_radius * 2)

    img_delimited = img_original.copy()
    if image.ndim == 2:  # If image is grayscale
        img_gray_inverted = image
    else:  # If image is RGB
        img_gray_inverted = np.bitwise_not(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
    circles_result = []
    last_circles = None

    # Improve probability to detect circles on border
    img_gray_inverted = draw_inner_border(img_gray_inverted, thickness=0.006, color=255)

    extra_margin_circle = 5
    # Initialize a blank mask to draw the detected circles.
    mask = np.zeros(img_gray_inverted.shape[:2], dtype=np.uint8)


    params = cv2.SimpleBlobDetector_Params()
    # Set up the parameters for the SimpleBlobDetector
    params.filterByCircularity = (minCircularity > 0)
    params.filterByConvexity = (minConvexity > 0)
    params.filterByInertia = (minInertiaRatio > 0)
    params.minDistBetweenBlobs = 1
    params.maxCircularity = 1.0
    params.maxConvexity = 1.0
    params.maxInertiaRatio = 1.0
    params.minThreshold = 10
    params.thresholdStep = 10
    params.minArea = minArea
    params.maxArea = maxArea

    # Run multiple detections to distinguish small circles from big circles
    for turn in range(6):
        params.maxThreshold = 250 - turn * 20
        if turn == 0:
            # First turn, very well defined circles
            params.minCircularity = 0.8
            params.minConvexity = 0.8
            params.minInertiaRatio = 0.7
        elif turn == 1:
            # Second turn, detect based on parameters set by user
            params.minCircularity = minCircularity
            params.minConvexity = minConvexity
            params.minInertiaRatio = minInertiaRatio
        elif turn == 2:
            # Third turn, detect smaller and imperfect circles
            params.minCircularity = minCircularity * 0.8
            params.minConvexity = minConvexity * 0.8
            params.minInertiaRatio = minInertiaRatio * 0.8
            params.minArea = int(minArea * 0.95)
            params.maxArea = int(maxArea * 0.8)
        elif turn == 3:
            # Fourth turn, even less perfect and smaller circles
            params.minCircularity = minCircularity * 0.7
            params.minConvexity = minConvexity * 0.7
            params.minInertiaRatio = minInertiaRatio * 0.6
            params.minArea = int(minArea * 0.90)
            params.maxArea = int(maxArea * 0.7)
        elif turn == 4:
            # Fifth turn, even less perfect and smaller circles
            params.minCircularity = minCircularity * 0.6
            params.minConvexity = minConvexity * 0.6
            params.minInertiaRatio = minInertiaRatio * 0.4
            params.minArea = int(minArea * 0.85)
            params.maxArea = int(maxArea * 0.6)
        elif turn == 5:
            # Sixth turn, last attempt
            debug_print("   Last attempt to detect circles.")


        # Validate parameters
        if params.maxArea < params.minArea:
            params.maxArea = params.minArea + 1

        # Create the blob detector with the specified parameters and detect blobs.
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(img_gray_inverted)

        # Process the detected keypoints to create a list of circles with their radius and draw them on the mask.
        circles = []
        for k in keypoints:
            x, y = int(k.pt[0]), int(k.pt[1])
            radius = int(k.size // 2 + extra_margin_circle)
            circles.append(((x, y), radius))

        circles = set_minimum_radius_circle(circles, min_radius)  # Make small circles bigger
        circles = set_maximum_radius_circle(circles, max_radius)  # Make big circles smaller
        debug_print(f"    Turn {turn}: {len(circles)} circles detected.")
        
        # Erase detected circles from the mask
        if last_circles is not None:
            # Reduces mask to tend to separate circles
            kernel_size = make_odd(min_radius // 2)
            img_gray_inverted = expand_mask_circle(img_gray_inverted, kernel_size=kernel_size, iterations=1)
            img_gray_inverted = erode_mask_circle(img_gray_inverted, kernel_size=kernel_size, iterations=1)
            img_gray_inverted = expand_mask_circle(img_gray_inverted, kernel_size=kernel_size-4, iterations=1)

            for circle in last_circles:
                (x, y), radius = circle
                cv2.circle(img_gray_inverted, (x, y), radius, 220 - turn * 20, thickness=-1)

            img_gray_inverted = blur_image(img_gray_inverted, kernel_size=13, sigmaX=0)

        #im_show_max(mask, window_name="Mask Final", max_resolution=400)
        last_circles = circles
        circles_result += circles
        
    debug_print(f"    {len(circles_result)} circles detected before remove overlap.")
    circles_result, discarded_circles = remove_overlapping_circles(circles_result, tolerance=tolerance_overlap)
    debug_print(f"    {len(circles_result)} circles detected after remove overlap.")

    # Draw discarded circles (gray)
    for circle in discarded_circles:
        (x, y), radius = circle
        cv2.circle(img_delimited, (x, y), radius + extra_margin_circle, (0, 0, 0), thickness=8)
        cv2.circle(img_delimited, (x, y), radius + extra_margin_circle, (200, 200, 200), thickness=3)

    # Draw circles after remove overlap
    color_increment = 200.0 / len(circles_result) if len(circles_result) != 0 else 0
    for i, circle in enumerate(circles_result):
        # The first detected is darker and the last is lighter
        color_based_on_position = (int(50 + color_increment * i), 80, 80)
        (x, y), radius = circle
        cv2.circle(img_delimited, (x, y), radius + extra_margin_circle, (0, 0, 0), thickness=12)
        cv2.circle(img_delimited, (x, y), radius + extra_margin_circle, color_based_on_position, thickness=4)
        cv2.circle(mask, (x, y), radius, 255, thickness=-1) 

    # Remove circles already detected in the last iteration
    for circle in last_circles:
        # Erase the circle from the mask
        (x, y), radius = circle
        cv2.circle(img_gray_inverted, (x, y), radius, 240, thickness=-1)
                
    return circles_result, img_delimited, img_gray_inverted

def check_overlap(circle1, circle2, overlap_level=0.8):
    """
    Checks if two circles overlap based on a specified overlap level.

    **Scenario:**
    When detecting multiple fruits, some detected circles might overlap due to proximity or segmentation errors. This function helps in identifying overlapping circles to avoid counting the same fruit multiple times.

    **Logic and Computer Vision Methods:**
    The function calculates the Euclidean distance between the centers of the two circles. It then checks if this distance is less than or equal to the sum of their radii multiplied by the overlap level. If so, the circles are considered overlapping based on the specified tolerance.

    Args:
        circle1 (tuple): A tuple representing the center coordinates and radius of the first circle ((x1, y1), r1).
        circle2 (tuple): A tuple representing the center coordinates and radius of the second circle ((x2, y2), r2).
        overlap_level (float): The required overlap level to consider the circles as overlapping. Default is 0.8.

    Returns:
        bool: True if the circles overlap with the required overlap level, False otherwise.
    """
    (x1, y1), r1 = circle1
    (x2, y2), r2 = circle2
    distance = ((x1 - x2)**2 + (y1 - y2)**2) ** 0.5
    return distance <= ((r1 + r2) * overlap_level)

def check_inside_or_overlap(circle1, circle2, tolerance=0.05):
    """
    Checks if two circles overlap or if one is inside the other.

    **Scenario:**
    In fruit detection, it's possible that some detected circles represent the same fruit due to segmentation overlaps. This function helps in determining whether two circles should be considered as representing the same fruit by checking for overlap or containment.

    **Logic and Computer Vision Methods:**
    The function first adjusts the radii of both circles based on the specified tolerance. It then calculates the distance between the centers and compares it to the sum of the adjusted radii. If the distance is less than the sum, or if one circle is entirely within the other, the circles are considered overlapping or one being inside the other.

    Args:
        circle1 (tuple): A tuple representing the center coordinates and radius of the first circle ((x1, y1), r1).
        circle2 (tuple): A tuple representing the center coordinates and radius of the second circle ((x2, y2), r2).
        tolerance (float): A tolerance level to consider the circles as overlapping. Default is 0.05 (5% of the radius).

    Returns:
        bool: True if the circles overlap or if one is inside the other, False otherwise.
    """
    (x1, y1), r1 = circle1
    (x2, y2), r2 = circle2

    r1 -= r1 * tolerance
    r2 -= r2 * tolerance

    # Calculate the distance between the centers of the two circles
    distance_centers = ((x1 - x2)**2 + (y1 - y2)**2) ** 0.5

    # Calculate the sum of the radii
    sum_radius = r1 + r2

    # Check if the circles are overlapping or if one is inside the other
    # This checks for any part of a circle's border being inside the other circle
    if distance_centers + min(r1, r2) < max(r1, r2) or distance_centers < sum_radius:
        return True
    else:
        return False

def remove_overlapping_circles(circles, tolerance=0.05):
    """
    Removes overlapping circles from a list of circles.

    **Scenario:**
    After detecting multiple circles representing fruits, some circles may overlap due to close proximity or detection inaccuracies. This function cleans the list by removing overlapping circles, ensuring each fruit is represented by a single, distinct circle.

    **Logic and Computer Vision Methods:**
    The function iterates through the list of circles and compares each pair using the `check_inside_or_overlap` function. If two circles are found to overlap beyond the specified tolerance, the latter circle is discarded to avoid duplication. This process continues until all overlapping circles are removed, resulting in a refined list of circles where each represents a unique fruit.

    Args:
        circles (list): A list of circles represented as tuples ((x, y), r), where (x, y) is the center and r is the radius.
        tolerance (float): The overlap tolerance to consider circles as overlapping.

    Returns:
        tuple: A tuple containing:
            - list: A new list of circles without the overlapping circles.
            - list: A list of discarded overlapping circles.
    """
    discarded_circles = []

    # Iterate over the circles
    i = 0
    while i < len(circles) - 1:
        # Iterate over the remaining circles
        for j in range(i + 1, len(circles)):
            # If the circles overlap, remove the overlapping circle
            if check_inside_or_overlap(circles[i], circles[j], tolerance=tolerance):
                # Remove the overlapping circle
                discarded_circles.append(circles[j])  # Keep the one inserted before
                del circles[j]
                i -= 1  # Reanalyze the same element again
                break
        i += 1

    return circles, discarded_circles

def set_minimum_radius_circle(circles, min_radius_circle):
    """
    Adjusts the radius of circles to ensure they meet a specified minimum radius.

    **Scenario:**
    During circle detection, some detected circles might be too small to accurately represent the fruits. This function ensures that all circles meet a minimum size requirement, preventing the detection of insignificant or noise-related circles.

    **Logic and Computer Vision Methods:**
    The function iterates through the list of circles and checks each circle's radius. If a circle's radius is smaller than the specified minimum, it is adjusted to the minimum value. This ensures consistency in the size of detected circles, which is important for uniform analysis and processing.

    Args:
        circles (list of tuples): A list where each tuple represents a circle with 
          its center coordinates ((x, y), radius).
        min_radius_circle (int): The minimum radius value that each circle's radius 
          should meet or exceed.

    Returns:
        list of tuples: A new list of circles with adjusted radii.
    """
    return [(center, radius if radius >= min_radius_circle else min_radius_circle) for center, radius in circles]

def set_maximum_radius_circle(circles, max_radius_circle):
    """
    Adjusts the radius of circles to ensure they do not exceed a specified maximum radius.

    **Scenario:**
    In scenarios where only fruits within a certain size range are of interest, this function ensures that no detected circle exceeds the maximum allowed size, thereby filtering out overly large detections that might be false positives or irrelevant.

    **Logic and Computer Vision Methods:**
    The function iterates through the list of circles and checks each circle's radius. If a circle's radius exceeds the specified maximum, it is adjusted down to the maximum value. This prevents the inclusion of excessively large circles in the final detection results.

    Args:
        circles (list of tuples): A list where each tuple represents a circle with 
          its center coordinates ((x, y), radius).
        max_radius_circle (int): The maximum radius value that each circle's radius 
          should not exceed.

    Returns:
        list of tuples: A new list of circles with adjusted radii.
    """
    return [(center, radius if radius <= max_radius_circle else max_radius_circle) for center, radius in circles]

def range_hue_and_normalize(array_channel, low_bound=10, high_bound=30):
    """
    Filters and normalizes hue values within a specified range in a hue channel image.

    **Scenario:**
    When processing the hue channel of an image for fruit detection, you might be interested in a specific range of hues that correspond to the fruit's color. This function isolates the desired hue range and normalizes the values to enhance contrast, making it easier to create accurate masks for detection.

    **Logic and Computer Vision Methods:**
    The function first ensures that the high bound is greater than the low bound. It then scales the low bound to match OpenCV's hue range. It filters the hue values to keep only those within the specified range and sets others to zero. The resulting values are then normalized to span the full 0-255 range using OpenCV's `normalize` function, enhancing the contrast of the selected hue range.

    Args:
        array_channel (numpy.ndarray): The hue channel of an image.
        low_bound (int): The lower bound of the hue range to keep.
        high_bound (int): The upper bound of the hue range to keep.

    Returns:
        numpy.ndarray: A normalized array where hue values within the range are scaled, and others are set to 0.
    """
    assert high_bound > low_bound, "high_bound must be greater than low_bound"
    low_bound = low_bound / 2  # Normalize to [0, 180]
    low_bound = low_bound / 2  # Normalize to [0, 180]

    # Step 1: Filter only values between low_bound and high_bound
    filtered_values = np.where((array_channel >= low_bound) & (array_channel <= high_bound), array_channel - low_bound, 0)
    normalized_values = cv2.normalize(filtered_values, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    normalized_values = normalized_values.reshape(array_channel.shape)

    return normalized_values

def detect_smooth_areas_rgb(image, kernel_size=21, threshold_value=15, noise=3, expand=3, it=1):
    """
    Detects smooth areas in an RGB image using a Gaussian filter.

    **Scenario:**
    In fruit detection, smooth areas might correspond to the background or regions without fruits. Identifying and masking these smooth areas can help focus the detection algorithms on the textured regions where fruits are likely to be present.

    **Logic and Computer Vision Methods:**
    The function applies a Gaussian blur to the image to create a smoothed version. It then converts both the original and blurred images to HSV color space and isolates the V channel. By comparing the original and blurred V channels, it calculates the intensity differences. Areas with low differences are considered smooth and are masked out. Morphological operations are then applied to clean the mask, ensuring that only significant smooth regions are detected.

    Args:
        image (numpy.ndarray): The input image in RGB color space.
        kernel_size (int): The kernel size for the Gaussian filter.
        threshold_value (int): The threshold value to detect smooth areas based on intensity difference.
        noise (int): The kernel size for salt-and-pepper noise removal.
        expand (int): The kernel size for mask dilation (expansion).
        it (int): The number of iterations for mask expansion.

    Returns:
        tuple: A tuple containing:
            - img_after (numpy.ndarray): The image after removing smooth areas.
            - mask (numpy.ndarray): The binary mask indicating smooth areas.
    """
    img_before = image.copy()
    # Smooth image to compare later
    kernel_size = make_odd(kernel_size)
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    blurred_image = cv2.cvtColor(blurred_image, cv2.COLOR_RGB2HSV)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Select only specific range of color
    channel = 2
    blurred_channel = range_hue_and_normalize(blurred_image[..., channel], low_bound=0, high_bound=100)
    img_channel = range_hue_and_normalize(image[..., channel], low_bound=0, high_bound=100)

    # Calculate the difference between the original and the smoothed image
    difference = cv2.absdiff(img_channel, blurred_channel)
    difference = np.clip(difference, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(difference, threshold_value, 255, cv2.THRESH_BINARY)
    if noise > 0:
        mask = remove_salt_and_pepper(mask, kernel_size=noise)
    if expand > 0:
        mask = expand_mask_circle(mask, kernel_size=expand, iterations=it)
    mask = ~mask

    img_after = cv2.bitwise_and(img_before, img_before, mask=mask)

    return img_after, mask
def draw_circles(image, circles, show_label=True, solid=False):
    """
    Draws circles and labels on an image.

    Args:
        image (numpy.ndarray): The image on which contours will be drawn.
        circles (list): List of tuples where each tuple contains center coordinates and radius, e.g., [((x, y), radius), ...].
        show_label (bool, optional): Indicates whether labels should be displayed. Default is True.
        solid (bool, optional): If True, draws solid circles; otherwise, draws outlined circles with multiple layers for a 3D effect. Default is False.

    Returns:
        numpy.ndarray: The image with drawn circles and labels.
    """
    # Create a copy of the original image to avoid modifying it directly.
    result_img = image.copy()

    # Sort the circles based on their coordinates using the sort_coordinates function.
    sorted_circles = sorted(circles, key=lambda circle: sort_coordinates(circle))

    # Iterate over each circle and its index.
    for i, circle in enumerate(sorted_circles):
        (x, y), radius = circle
        if solid:
            # Draw a solid circle with increased radius for better visibility.
            cv2.circle(result_img, (x, y), radius + 5, (230, 230, 230), thickness=-1)
        else:
            # Draw multiple concentric circles with varying thickness to create a layered effect.
            cv2.circle(result_img, (x, y), radius + 2, (20, 20, 20), thickness=8)
            cv2.circle(result_img, (x, y), radius + 2, (210, 210, 210), thickness=6)
            cv2.circle(result_img, (x, y), radius + 2, (250, 250, 250), thickness=4)

            if show_label:
                # Create a label for the circle based on its index.
                label = f"{i + 1}"
                # Determine the size of the text to center it on the circle.
                (text_width, text_height), _ = cv2.getTextSize(
                    label,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.7,
                    thickness=2
                )

                # Calculate the position to place the text so that it's centered.
                text_position = (int(x - text_width / 2), int(y + text_height / 2))

                # Draw the label text on the image.
                cv2.putText(
                    result_img,
                    label,
                    text_position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.7,
                    color=(40, 40, 40),
                    thickness=2
                )

    # Return the image with drawn circles and labels.
    return result_img


def get_exif_data(image):
    """
    Extracts GPS coordinates and camera information from an image's EXIF metadata.

    Args:
        image (PIL.Image.Image): The image from which to obtain EXIF metadata.

    Returns:
        tuple: A tuple containing GPS coordinates (latitude, longitude) and camera information 
               (manufacturer, model, capture_date). If data is unavailable, returns (None, None).
    """
    exif_data = {}
    try:
        # Retrieve EXIF information from the image.
        if info := image._getexif():
            debug_print(f'EXIF={info}')
            for tag, value in info.items():
                decoded = EXIF.TAGS.get(tag, tag)
                if decoded == "GPSInfo":
                    gps_data = {}
                    for t in value:
                        sub_decoded = EXIF.GPSTAGS.get(t, t)
                        gps_data[sub_decoded] = value[t]
                    exif_data[decoded] = gps_data
                else:
                    exif_data[decoded] = value
        # Extract GPS coordinates and camera information.
        return get_gps_coordinates(exif_data), get_camera_info_from_exif(exif_data)
    except Exception as e:
        # Handle any errors during EXIF extraction.
        debug_print(f"Error reading EXIF data: {e}")
        return (None, None), (None, None, None)


def get_camera_info_from_exif(exif_data):
    """
    Extracts camera manufacturer, model, and capture date from EXIF metadata.

    Args:
        exif_data (dict): The EXIF metadata of the image.

    Returns:
        tuple: A tuple containing the camera manufacturer, camera model, and capture date.
    """
    camera_manufacturer = exif_data.get('Make')
    camera_model = exif_data.get('Model')
    capture_date = exif_data.get('DateTimeOriginal') or exif_data.get('DateTime')

    return camera_manufacturer, camera_model, capture_date


def convert_to_degrees(value):
    """
    Converts GPS coordinates from degrees, minutes, and seconds to decimal degrees.

    Args:
        value (tuple): The GPS coordinates in the format (degrees, minutes, seconds).

    Returns:
        float: The GPS coordinates converted to decimal degrees.
    """
    d, m, s = value
    return d + (m / 60.0) + (s / 3600.0)


def get_gps_coordinates(exif_data):
    """
    Retrieves GPS coordinates from EXIF metadata if available.

    Args:
        exif_data (dict): The EXIF metadata of the image.

    Returns:
        tuple: A tuple containing the GPS coordinates in decimal degrees (latitude, longitude).
               If GPS coordinates are not available, returns (None, None).
    """
    gps_info = exif_data.get("GPSInfo")
    if not gps_info:
        return None, None

    latitude = gps_info.get("GPSLatitude")
    latitude_ref = gps_info.get("GPSLatitudeRef")
    longitude = gps_info.get("GPSLongitude")
    longitude_ref = gps_info.get("GPSLongitudeRef")

    if latitude and longitude and latitude_ref and longitude_ref:
        # Convert latitude and longitude to decimal degrees.
        latitude = convert_to_degrees(latitude)
        if latitude_ref != "N":
            latitude = -latitude
        longitude = convert_to_degrees(longitude)
        if longitude_ref != "E":
            longitude = -longitude
        return latitude, longitude
    else:
        return None, None


def extract_metadata_EXIF(img_IO, result):
    """
    Extracts useful information from an image's EXIF metadata, such as GPS coordinates, 
    capture date, and camera details.

    Args:
        img_IO (PIL.Image.Image): The input image object.
        result (dict): A dictionary to store the extracted metadata.

    Returns:
        bool: True if metadata was found and stored successfully, False otherwise.
    """
    # Extract GPS coordinates and camera information from EXIF data.
    (lat, lon), (manufacturer, mobile_model, capture_date) = get_exif_data(img_IO)

    if lat and lon:
        # Store GPS coordinates and capture date in the result dictionary.
        result['coordinates'] = (round(lat, 6), round(lon, 6))
        result['capture_date'] = capture_date

    # Handle DJI drones' metadata bug by removing null characters from model and manufacturer names.
    if mobile_model:
        result['mobile_model'] = mobile_model.replace("\u0000", "")

    if manufacturer:
        result['mobile_manufacturer'] = manufacturer.replace("\u0000", "")

    return True


def kmeans_recolor(original_image, n_clusters=8):
    """
    Recolors an image using K-Means clustering to identify dominant colors.

    **Scenario:**
    In a fruit detection system, identifying dominant colors can help in classifying different types of fruits based on their color profiles.

    **Logic and Computer Vision Methods:**
    The function downsizes the image for faster processing, flattens the image data, and applies K-Means clustering to group pixel colors into specified clusters. It then recolors the image based on the cluster centers, effectively reducing the color palette to the most dominant colors. Additionally, it applies a color map for visualization purposes.

    Args:
        original_image (numpy.ndarray): The original image to be processed.
        n_clusters (int): The number of color clusters to form.

    Returns:
        tuple: A tuple containing the recolored image, the color-mapped image, and the cluster labels.
    """
    # Calculate the image's dimensions.
    height, width = original_image.shape[:2]

    scaling_factor = 0.5

    # Calculate the new dimensions based on the scaling factor.
    new_height, new_width = int(height * scaling_factor), int(width * scaling_factor)

    # Resize the image to speed up K-Means processing.
    original_image = cv2.resize(original_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Flatten the image for K-Means clustering.
    original_vectorized = original_image.reshape((-1, 3))
    original_vectorized = np.float32(original_vectorized)
    
    # Define criteria and apply K-Means clustering.
    debug_print("Applying KMeans")
    attempts = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    rets, labels, centers = cv2.kmeans(
        original_vectorized, 
        n_clusters, 
        None, 
        criteria,
        attempts,
        cv2.KMEANS_PP_CENTERS
    )

    # Reshape labels to match the resized image's dimensions.
    clustered_labels = labels.reshape((original_image.shape[0], original_image.shape[1]))
    clustered_labels = np.uint8(clustered_labels)

    debug_print("Coloring Viridis")
    # Normalize labels for applying a color map.
    labels_8bit = cv2.normalize(
        clustered_labels, 
        None, 
        alpha=0, 
        beta=255, 
        norm_type=cv2.NORM_MINMAX, 
        dtype=cv2.CV_8U
    )

    # Apply a color map (e.g., COLORMAP_JET) for visualization.
    clustered_rgb = cv2.applyColorMap(labels_8bit, cv2.COLORMAP_JET)
    
    debug_print("Recoloring based on original")
    # Assign each pixel the color of its corresponding cluster center.
    center = np.uint8(centers)
    res = center[labels.flatten()]
    recolored_image = res.reshape((original_image.shape))
    debug_print("KMeans finished")

    # Resize the recolored image back to the original size.
    recolored_image = cv2.resize(recolored_image, (width, height), interpolation=cv2.INTER_AREA)
    clustered_rgb = cv2.resize(clustered_rgb, (width, height), interpolation=cv2.INTER_AREA)
    clustered_labels = cv2.resize(clustered_labels, (width, height), interpolation=cv2.INTER_AREA)

    return recolored_image, clustered_rgb, clustered_labels


def fill_holes_grayscale(image):
    """
    Fills holes in a binary grayscale image to create a solid mask.

    **Scenario:**
    When detecting fruits, the binary mask might have small holes or gaps due to shadows or occlusions. Filling these holes ensures a complete representation of each fruit.

    **Logic and Computer Vision Methods:**
    The function thresholds the image to create a binary mask, applies hole filling using `binary_fill_holes`, inverts the mask, and combines it with the original image to fill the holes.

    Args:
        image (numpy.ndarray): The binary grayscale image.

    Returns:
        numpy.ndarray: The image after holes have been filled.
    """
    # Threshold the grayscale image to create a binary mask.
    _, binary_mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Fill holes in the binary mask using binary_fill_holes.
    filled_mask = ndi.binary_fill_holes(binary_mask).astype(np.uint8) * 255
    
    # Invert the mask to match the original image's background.
    filled_mask_inv = cv2.bitwise_not(filled_mask)
    
    # Combine the filled mask with the original image.
    result = cv2.bitwise_or(image, image, mask=filled_mask_inv)
    
    return result


def fill_holes_with_gray(image, new_pixel=65):
    """
    Fills holes in a binary image with a specified grayscale value.

    **Scenario:**
    In scenarios where certain regions (like the background) need to be filled with a uniform color after segmentation, this function replaces hole regions with a specified grayscale value.

    **Logic and Computer Vision Methods:**
    The function thresholds the image, fills holes using `binary_fill_holes`, identifies the hole regions, and replaces those regions with a new grayscale value using NumPy's `where` function.

    Args:
        image (numpy.ndarray): The binary grayscale image.
        new_pixel (int, optional): The grayscale value to assign to filled holes. Default is 65.

    Returns:
        numpy.ndarray: The image after holes have been filled with the specified grayscale value.
    """
    # Threshold the grayscale image to create a binary mask.
    _, binary_mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Fill holes in the binary mask using binary_fill_holes.
    filled_mask = ndi.binary_fill_holes(binary_mask).astype(np.uint8) * 255
    holes_mask = filled_mask - binary_mask

    # Replace hole regions with the specified grayscale value.
    result = np.where(holes_mask == 255, new_pixel, image)

    return result


def average_excluding_value(image, value=255):
    """
    Calculates the average pixel value of an image excluding a specified value.

    **Scenario:**
    When processing segmented images, you might want to calculate statistics only for the foreground objects, excluding the background pixels marked by a specific value (e.g., 255).

    **Logic and Computer Vision Methods:**
    The function flattens the image array, filters out pixels matching the specified value, and computes the mean of the remaining pixels.

    Args:
        image (numpy.ndarray): The input image.
        value (int, optional): The pixel value to exclude from the average calculation. Default is 255.

    Returns:
        int: The average pixel value excluding the specified value.
    """
    # Flatten the image array to 1D for easier processing.
    flat_image = image.flatten()
    
    # Filter out pixels with the specified value.
    filtered_pixels = flat_image[flat_image != value]
    
    # Calculate the average value of the remaining pixels.
    if len(filtered_pixels) > 0:
        average_value = np.mean(filtered_pixels).astype(np.uint8)
    else:
        average_value = 255
    
    return average_value


def draw_inner_border(image, thickness=0.01, color=(255, 255, 255)):
    """
    Draws an inner border inside an image without altering its dimensions.

    **Scenario:**
    When preparing images for object detection, adding an inner border can help in highlighting the detection area or separating objects from the edges.

    **Logic and Computer Vision Methods:**
    The function calculates the border thickness based on the image's dimensions, then applies the specified color to the top, bottom, left, and right edges of the image.

    Args:
        image (numpy.ndarray): The image on which to draw the border.
        thickness (float, optional): The thickness of the border as a fraction of the image's smaller dimension. Default is 0.01.
        color (tuple, optional): The color of the border in BGR format. Default is white.

    Returns:
        numpy.ndarray: The image with the inner border drawn.
    """
    # Ensure the border thickness is at least 1 pixel.
    thickness = int(thickness * min(image.shape[:2])) + 1

    # Apply the specified color as a border within the image.
    image[0:thickness, :] = image[-thickness:, :] = image[:, 0:thickness] = image[:, -thickness:] = color

    return image


def save_image(image, path_filename_result):
    """
    Saves an image to the specified file path after converting it from BGR to RGB color space.

    **Scenario:**
    After processing and annotating images (e.g., drawing detected fruit circles), you need to save the results for later review or reporting.

    **Logic and Computer Vision Methods:**
    OpenCV uses BGR color space by default. Before saving, the image is converted to RGB to ensure colors are preserved correctly. The `imwrite` function then saves the image to the desired path.

    Args:
        image (numpy.ndarray): The image to be saved, in BGR color space.
        path_filename_result (str): The file path where the image will be saved.

    Returns:
        None
    """
    # Convert the image from BGR to RGB color space.
    img_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Save the image to the specified path.
    cv2.imwrite(path_filename_result, img_RGB)
    debug_print(f"    Saved: {path_filename_result}")


def crop_image_to_aspect_ratio(image, ratio=(4, 3)):
    """
    Crops an image to a specified aspect ratio based on its orientation.

    **Scenario:**
    In fruit detection, maintaining a consistent aspect ratio can be important for feeding images into machine learning models or for standardized reporting.

    **Logic and Computer Vision Methods:**
    The function determines whether the image is wider or taller, calculates the desired aspect ratio (4:3 or 3:4), and then crops the image centrally to match the target aspect ratio.

    Args:
        image (numpy.ndarray): The input image to be cropped.
        ratio (tuple, optional): The desired aspect ratio as a tuple (width, height). Default is (4, 3).

    Returns:
        numpy.ndarray: The cropped image with the specified aspect ratio.
    """
    # Get the dimensions of the image.
    height, width, _ = image.shape
    
    # Calculate the aspect ratio of the image.
    aspect_ratio = width / height
    
    # Determine the desired aspect ratio based on the image's orientation.
    if aspect_ratio > 1:
        # Image is wider than tall, so crop to 4:3 aspect ratio.
        desired_aspect_ratio = ratio[0] / ratio[1]
    else:
        # Image is taller than wide, so crop to 3:4 aspect ratio.
        desired_aspect_ratio = ratio[1] / ratio[0]
    
    # Calculate the new dimensions for the cropped image.
    if aspect_ratio > desired_aspect_ratio:
        # Crop the width.
        new_width = int(height * desired_aspect_ratio)
        x_start = (width - new_width) // 2
        x_end = x_start + new_width
        y_start = 0
        y_end = height
    else:
        # Crop the height.
        new_height = int(width / desired_aspect_ratio)
        x_start = 0
        x_end = width
        y_start = (height - new_height) // 2
        y_end = y_start + new_height
    
    # Crop the image.
    cropped_image = image[y_start:y_end, x_start:x_end]
    
    return cropped_image


def smooth_color(image, kernel_size=21, min_brightness=10, max_brightness=210):
    """
    Smooths the color of an image by applying Gaussian blur and adjusting brightness in LAB color space.

    **Scenario:**
    In fruit detection, smoothing colors can help reduce noise and highlight the true color regions of fruits, making detection more accurate.

    **Logic and Computer Vision Methods:**
    The function converts the image to LAB color space, clamps the brightness values within a specified range, applies Gaussian blur to the A and B channels to smooth colors, and then converts the image back to RGB color space.

    Args:
        image (numpy.ndarray): The input image in RGB color space.
        kernel_size (int, optional): The size of the Gaussian kernel. Must be odd. Default is 21.
        min_brightness (int, optional): The minimum brightness value. Default is 10.
        max_brightness (int, optional): The maximum brightness value. Default is 210.

    Returns:
        numpy.ndarray: The color-smoothed image in RGB color space.
    """
    # Convert the image from RGB to LAB color space.
    image_LAB = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.uint8)
    new_image = image.copy()

    # Extract the L (lightness), A, and B channels.
    brightness = image_LAB[..., 0]
    color_A = image_LAB[..., 1]
    color_B = image_LAB[..., 2]

    # Clamp the brightness values to remove extreme lighting conditions.
    brightness = np.clip(brightness, min_brightness, max_brightness).astype(np.uint8)

    # Apply Gaussian blur to the A and B channels to smooth color variations.
    color_A = cv2.GaussianBlur(color_A, ksize=(kernel_size, kernel_size), sigmaX=0)
    color_B = cv2.GaussianBlur(color_B, ksize=(kernel_size, kernel_size), sigmaX=0)

    # Assign the modified channels back to the LAB image.
    new_image[..., 0] = brightness
    new_image[..., 1] = color_A
    new_image[..., 2] = color_B

    # Convert the LAB image back to RGB color space.
    new_image = cv2.cvtColor(new_image, cv2.COLOR_LAB2RGB).astype(np.uint8)

    return new_image


def tunning_blur(image, clahe_grid=3, clahe_limit=1.5, salt_pepper=3, blur_size=3):
    """
    Provides an interactive tool to tune image blurring parameters, including CLAHE, noise removal, and Gaussian blur.

    **Scenario:**
    When preparing images for fruit detection, adjusting blurring parameters can help in enhancing features and reducing noise, leading to more accurate detections.

    **Logic and Computer Vision Methods:**
    The function creates GUI windows with trackbars to adjust parameters like CLAHE grid size and limit, salt-and-pepper noise kernel size, and Gaussian blur kernel size. As parameters are adjusted, the image is processed in real-time, and the results are displayed for immediate feedback.

    Args:
        image (numpy.ndarray): The input image to be processed.
        clahe_grid (int, optional): The grid size for CLAHE. Default is 3.
        clahe_limit (float, optional): The clip limit for CLAHE. Default is 1.5.
        salt_pepper (int, optional): The kernel size for salt-and-pepper noise removal. Default is 3.
        blur_size (int, optional): The kernel size for Gaussian blur. Default is 3.

    Returns:
        tuple: A tuple containing the processed image and the final parameter values.
    """
    # Resize the image for easier viewing.
    img_before = resize_image(image, max_resolution=800)

    # Create GUI windows for before and after images, and for parameter configuration.
    cv2.destroyAllWindows()
    cv2.namedWindow('Before Blur')
    cv2.moveWindow('Before Blur', 0, 0)

    cv2.namedWindow('After Blur')
    cv2.moveWindow('After Blur', 0, 525)

    cv2.namedWindow('Config Blur')
    cv2.moveWindow('Config Blur', 0, 1200)
    cv2.resizeWindow('Config Blur', 700, 160)

    global update
    update = False

    # Callback function for trackbar events.
    def update_parameters(x):
        global update
        update = True

    # Create trackbars for adjusting CLAHE, noise removal, and blur parameters.
    cv2.createTrackbar('grid', 'Config Blur', clahe_grid, 29, update_parameters)
    cv2.createTrackbar('limit', 'Config Blur', int(clahe_limit * 10), 79, update_parameters)
    cv2.createTrackbar('noise', 'Config Blur', salt_pepper, 29, update_parameters)
    cv2.createTrackbar('blur', 'Config Blur', blur_size, 29, update_parameters)

    # Display the original image.
    im_show_max(img_before, window_name='Before Blur', max_resolution=600)

    while True:
        if update:
            # Retrieve and adjust parameters from trackbars.
            clahe_grid = cv2.getTrackbarPos('grid', 'Config Blur')
            clahe_limit = cv2.getTrackbarPos('limit', 'Config Blur') / 10.0 
            salt_pepper = make_odd(cv2.getTrackbarPos('noise', 'Config Blur'))
            blur_size = make_odd(max(1, cv2.getTrackbarPos('blur', 'Config Blur')))

            img_tmp = img_before.copy()
            # Apply CLAHE if grid size and limit are set.
            if clahe_grid > 0 and clahe_limit > 0:
                _, img_tmp, _ = apply_clahe(
                    img_tmp, 
                    tileGridSize=(clahe_grid, clahe_grid), 
                    clipLimit=clahe_limit
                )
            # Remove salt-and-pepper noise if specified.
            if salt_pepper > 0:
                img_tmp = remove_salt_and_pepper(img_tmp, kernel_size=salt_pepper)
            # Apply Gaussian blur if specified.
            if blur_size > 0:
                img_tmp = blur_image(img_tmp, kernel_size=blur_size)

            # Display the processed image.
            im_show_max(img_tmp, window_name='After Blur', max_resolution=800)
            # Keep the configuration window on top.
            cv2.setWindowProperty('Config Blur', cv2.WND_PROP_TOPMOST, 1)
            update = False

        # Exit the loop when 'q' is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Close all windows.
    cv2.destroyWindow("Before Blur")
    cv2.destroyWindow("After Blur")
    cv2.destroyWindow("Config Blur")

    return img_tmp, clahe_grid, clahe_limit, salt_pepper, blur_size


def tunning_color(image, parameters=[0, 20, 20, 360, 100, 100, 5, 5, 5, 1], window_name="", color_space='HSV'):
    """
    Provides an interactive tool to tune color filtering parameters in HSV or LAB color spaces.

    **Scenario:**
    When detecting fruits, adjusting color filtering parameters can help in isolating specific fruit colors from the background, improving detection accuracy.

    **Logic and Computer Vision Methods:**
    The function creates GUI windows with trackbars to adjust color range parameters based on the chosen color space (HSV or LAB). As parameters are adjusted, the image is filtered in real-time to show the selected and discarded areas, allowing for precise tuning of color thresholds.

    Args:
        image (numpy.ndarray): The input image to be processed.
        parameters (list, optional): A list of initial parameter values. Default is [0, 20, 20, 360, 100, 100, 5, 5, 5, 1].
        window_name (str, optional): The name of the window for display. Default is an empty string.
        color_space (str, optional): The color space to use ('HSV' or 'LAB'). Default is 'HSV'.

    Returns:
        tuple: A tuple containing the selected area, discarded area, and the final mask.
    """
    # Ensure the selected color space is valid.
    assert color_space in ['HSV', 'LAB'], "Color space must be 'HSV' or 'LAB'."
    
    img_before = image.copy()

    # Define window names based on the provided window_name.
    window_before = f'Before - Color {window_name}'
    window_after = f'After - Color {window_name}'
    window_config = f'Config - Color {window_name}'

    # Create GUI windows for before and after images, and for parameter configuration.
    cv2.destroyAllWindows()
    cv2.namedWindow(window_before)
    cv2.moveWindow(window_before, 0, 0)
    cv2.namedWindow(window_after)
    cv2.moveWindow(window_after, 0, 420)
    cv2.namedWindow(window_config)
    cv2.moveWindow(window_config, 0, 1100)
    cv2.resizeWindow(window_config, 700, 380)
    global update
    update = False

    # Callback function for trackbar events.
    def update_parameters(x):
        global update
        update = True

    # Create trackbars for adjusting color range parameters based on the selected color space.
    if color_space == 'HSV':
        # HSV format: Hue=[0:360], Saturation=[0:100], Brightness=[0:100].
        cv2.createTrackbar('H min', window_config, parameters[0], 359, update_parameters)
        cv2.createTrackbar('H max', window_config, parameters[3], 359, update_parameters)
        cv2.createTrackbar('S min', window_config, parameters[1], 100, update_parameters)
        cv2.createTrackbar('S max', window_config, parameters[4], 100, update_parameters)
        cv2.createTrackbar('V min', window_config, parameters[2], 100, update_parameters)
        cv2.createTrackbar('V max', window_config, parameters[5], 100, update_parameters)
    elif color_space == 'LAB':
        # LAB format: L=[0:100], A=[-128:127], B=[-128:127].
        cv2.createTrackbar('L min', window_config, parameters[0], 100, update_parameters)
        cv2.createTrackbar('L max', window_config, parameters[3], 100, update_parameters)
        cv2.createTrackbar('A(GR) min', window_config, parameters[1] + 128, 255, update_parameters)
        cv2.createTrackbar('A(GR) max', window_config, parameters[4] + 128, 255, update_parameters)
        cv2.createTrackbar('B(BY) min', window_config, parameters[2] + 128, 255, update_parameters)
        cv2.createTrackbar('B(BY) max', window_config, parameters[5] + 128, 255, update_parameters)
    # Additional trackbars for noise removal, mask expansion, and closing operations.
    cv2.createTrackbar('Noise', window_config, parameters[6], 90, update_parameters)
    cv2.createTrackbar('Expand', window_config, parameters[7], 90, update_parameters)
    cv2.createTrackbar('Close', window_config, parameters[8], 90, update_parameters)
    cv2.createTrackbar('Iter.', window_config, parameters[9], 10, update_parameters)
    
    while True:
        if update:
            # Retrieve current positions of the trackbars.
            if color_space == 'HSV':
                parameters[0] = cv2.getTrackbarPos('H min', window_config)
                parameters[1] = cv2.getTrackbarPos('S min', window_config)
                parameters[2] = cv2.getTrackbarPos('V min', window_config)
                parameters[3] = cv2.getTrackbarPos('H max', window_config)
                parameters[4] = cv2.getTrackbarPos('S max', window_config)
                parameters[5] = cv2.getTrackbarPos('V max', window_config)
            elif color_space == 'LAB':
                parameters[0] = cv2.getTrackbarPos('L min', window_config)
                parameters[1] = cv2.getTrackbarPos('A(GR) min', window_config) - 128
                parameters[2] = cv2.getTrackbarPos('B(BY) min', window_config) - 128
                parameters[3] = cv2.getTrackbarPos('L max', window_config)
                parameters[4] = cv2.getTrackbarPos('A(GR) max', window_config) - 128
                parameters[5] = cv2.getTrackbarPos('B(BY) max', window_config) - 128

            # Retrieve additional parameters for noise removal, mask expansion, and closing.
            parameters[6] = make_odd(cv2.getTrackbarPos('Noise', window_config))
            parameters[7] = cv2.getTrackbarPos('Expand', window_config)
            parameters[8] = make_odd(cv2.getTrackbarPos('Close', window_config))
            parameters[9] = cv2.getTrackbarPos('Iter.', window_config)

            # Apply color filtering based on the adjusted parameters.
            if parameters[6] > 0:
                selected_area, discarded_area, mask_tmp = filter_color(
                    img_before, 
                    color_ini=parameters[0:3], 
                    color_end=parameters[3:6], 
                    noise=parameters[6],
                    expand=parameters[7],
                    close=parameters[8],
                    iterations=parameters[9],
                    color_space=color_space
                )

            # Display the original and discarded areas side by side.
            im_show_max(
                build_mosaic([img_before, discarded_area], mosaic_dims=(1, 2)),
                window_name=window_before,
                max_resolution=800
            )
            # Display the selected area after filtering.
            im_show_max(selected_area, window_name=window_after, max_resolution=800)
            
            # Keep the configuration window on top.
            cv2.setWindowProperty(window_config, cv2.WND_PROP_TOPMOST, 1)
            update = False

        # Exit the loop when 'q' is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Close all windows.
    cv2.destroyWindow(window_before)
    cv2.destroyWindow(window_after)
    cv2.destroyWindow(window_config)

    return selected_area, discarded_area, mask_tmp


def filter_shadow(img, brightness=127, exposure=127, contrast=127, highlights=0, shadows=0, saturation=127):
    """
    Adjusts image brightness, exposure, contrast, highlights, shadows, and saturation to reduce shadows.

    **Scenario:**
    In fruit detection, shadows can obscure details and affect color-based segmentation. Adjusting these parameters helps in minimizing shadow effects, making fruits more distinguishable.

    **Logic and Computer Vision Methods:**
    The function scales brightness, exposure, and contrast, clamps brightness to a specified range, adjusts saturation, and manipulates the brightness channel to reduce shadows and highlights using conditional operations.

    Args:
        img (numpy.ndarray): The input image in BGR color space.
        brightness (int, optional): Brightness adjustment value. Default is 127.
        exposure (int, optional): Exposure adjustment value. Default is 127.
        contrast (int, optional): Contrast adjustment value. Default is 127.
        highlights (int, optional): Highlights adjustment value. Default is 0.
        shadows (int, optional): Shadows adjustment value. Default is 0.
        saturation (int, optional): Saturation adjustment value. Default is 127.

    Returns:
        numpy.ndarray: The image after shadow and highlight adjustments.
    """
    # Convert the image from BGR to HSV color space.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Adjust brightness and exposure using a linear transformation.
    brightness_scale = brightness - 127   # Range from -127 to 128
    exposure_scale = exposure - 127       # Range from -127 to 128
    contrast_scale = (contrast - 127) / 127 + 1  # Scale from about 0.0 to 2.0
    saturation_scale = (saturation - 127) / 127 + 1  # Scale from about 0.0 to 2.0
    shadow_scale = shadows / 50.0
    highlight_scale = highlights / 50.0

    # Apply brightness and exposure adjustments.
    img_adjusted = cv2.convertScaleAbs(v, alpha=contrast_scale, beta=brightness_scale + exposure_scale)
    
    # Adjust saturation.
    s = cv2.convertScaleAbs(s, alpha=saturation_scale)
    
    # Adjust shadows by darkening areas below a certain brightness threshold.
    threshold_shadows = 150.0
    v = np.where(
        v < threshold_shadows, 
        np.clip(v * (1.0 + shadow_scale * ((threshold_shadows - v) / threshold_shadows)), 0.0, threshold_shadows), 
        v
    )

    # Adjust highlights by reducing brightness in bright areas.
    threshold_highlights = 255.0 - 150.0
    v = np.where(
        v > threshold_highlights, 
        np.clip(v * (1.0 - highlight_scale * ((v - threshold_highlights) / (255.0 - threshold_highlights))), threshold_highlights, 255.0), 
        v
    )
    
    # Ensure the brightness channel is of type uint8.
    v = v.astype(np.uint8)
    
    # Merge the adjusted channels back into the HSV image.
    hsv_adjusted = cv2.merge([h, s, v])
    # Convert the HSV image back to BGR color space.
    result = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)
    
    return result


def tunning_shadow(image, brightness=127, exposure=127, contrast=127, highlights=0, shadows=0, saturation=127):
    """
    Provides an interactive tool to tune shadow and highlight adjustment parameters.

    **Scenario:**
    When preparing images for fruit detection, interactively adjusting shadow and highlight parameters can help in minimizing the impact of uneven lighting, ensuring fruits are clearly visible.

    **Logic and Computer Vision Methods:**
    The function creates GUI windows with trackbars for adjusting brightness, exposure, contrast, highlights, shadows, and saturation. As parameters are adjusted, the image is processed in real-time to show the effects of these adjustments, allowing for precise tuning.

    Args:
        image (numpy.ndarray): The input image to be processed.
        brightness (int, optional): Initial brightness value. Default is 127.
        exposure (int, optional): Initial exposure value. Default is 127.
        contrast (int, optional): Initial contrast value. Default is 127.
        highlights (int, optional): Initial highlights adjustment. Default is 0.
        shadows (int, optional): Initial shadows adjustment. Default is 0.
        saturation (int, optional): Initial saturation value. Default is 127.

    Returns:
        tuple: A tuple containing the processed image and the final parameter values.
    """
    # Make a copy of the original image for processing.
    img_before = image.copy()

    # Define window names for before and after images, and for configuration.
    window_before = 'Before - Shadow Adjustments'
    window_after = 'After - Shadow Adjustments'
    window_config = 'Config - Shadow Adjustments'

    # Create GUI windows.
    cv2.destroyAllWindows()
    cv2.namedWindow(window_before)
    cv2.moveWindow(window_before, 0, 0)
    cv2.namedWindow(window_after)
    cv2.moveWindow(window_after, 0, 420)
    cv2.namedWindow(window_config)
    cv2.moveWindow(window_config, 0, 1100)
    cv2.resizeWindow(window_config, 700, 380)
    global update
    update = False

    # Callback function for trackbar events.
    def update_parameters(x):
        global update
        update = True

    # Create trackbars for adjusting brightness, exposure, contrast, highlights, shadows, and saturation.
    cv2.createTrackbar('Brightness', window_config, brightness, 255, update_parameters)
    cv2.createTrackbar('Exposure', window_config, exposure, 255, update_parameters)
    cv2.createTrackbar('Contrast', window_config, contrast, 255, update_parameters)
    cv2.createTrackbar('Highlights', window_config, highlights, 100, update_parameters)
    cv2.createTrackbar('Shadows', window_config, shadows, 100, update_parameters)
    cv2.createTrackbar('Saturation', window_config, saturation, 255, update_parameters)
    
    while True:
        if update:
            # Retrieve current positions of the trackbars.
            brightness = cv2.getTrackbarPos('Brightness', window_config)
            exposure = cv2.getTrackbarPos('Exposure', window_config)
            contrast = cv2.getTrackbarPos('Contrast', window_config)
            highlights = cv2.getTrackbarPos('Highlights', window_config)
            shadows = cv2.getTrackbarPos('Shadows', window_config)
            saturation = cv2.getTrackbarPos('Saturation', window_config)

            # Apply shadow and highlight adjustments based on the current parameters.
            img_after = filter_shadow(
                img_before, 
                brightness=brightness, 
                exposure=exposure, 
                contrast=contrast, 
                highlights=highlights, 
                shadows=shadows, 
                saturation=saturation
            )

            # Display the original and adjusted images side by side.
            im_show_max(
                build_mosaic([img_before, img_after], mosaic_dims=(1, 2)), 
                window_name=window_before, 
                max_resolution=800
            )
            # Display the adjusted image separately.
            im_show_max(img_after, window_name=window_after, max_resolution=800)

            # Keep the configuration window on top.
            cv2.setWindowProperty(window_config, cv2.WND_PROP_TOPMOST, 1)
            update = False

        # Exit the loop when 'q' is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Close all windows.
    cv2.destroyWindow("Before Shadow Adjustments")
    cv2.destroyWindow("After Shadow Adjustments")
    cv2.destroyWindow("Config - Shadow Adjustments")

    return img_after, brightness, exposure, contrast, highlights, shadows, saturation


def tunning_GrabCut_mask(mask_base, close_PR_FGD, erode_PR_FGD, erode_FGD, noise_FGD, erode_BGD):
    """
    Provides an interactive tool to tune GrabCut mask parameters for foreground and background segmentation.

    **Scenario:**
    When accurately segmenting fruits from the background, fine-tuning GrabCut parameters helps in achieving precise boundaries, especially in complex scenes with overlapping objects or varying lighting.

    **Logic and Computer Vision Methods:**
    The function creates GUI windows with trackbars for adjusting parameters related to morphological operations on the GrabCut mask. As parameters are adjusted, the mask is processed in real-time, showing the effects of closing, erosion, and noise removal on the segmentation.

    Args:
        mask_base (numpy.ndarray): The initial mask for GrabCut segmentation.
        close_PR_FGD (int): Kernel size for morphological closing on probable foreground.
        erode_PR_FGD (int): Kernel size for erosion on probable foreground.
        erode_FGD (int): Kernel size for erosion on definite foreground.
        noise_FGD (int): Kernel size for noise removal in definite foreground.
        erode_BGD (int): Kernel size for erosion on background.

    Returns:
        tuple: A tuple containing the refined GrabCut mask, color-mapped mask, and the final parameter values.
    """
    # Make a copy of the original image and mask for processing.
    img_before = mask_base.copy()

    # Define window names for before and after masks, and for configuration.
    window_before = 'Before GrabCut'
    window_after = 'After GrabCut'
    window_config = 'Config GrabCut'

    # Create GUI windows.
    cv2.destroyAllWindows()
    cv2.namedWindow(window_before)
    cv2.moveWindow(window_before, 0, 0)
    cv2.namedWindow(window_after)
    cv2.moveWindow(window_after, 0, 420)
    cv2.namedWindow(window_config)
    cv2.moveWindow(window_config, 0, 1100)
    cv2.resizeWindow(window_config, 700, 210)
    global update
    update = False

    # Callback function for trackbar events.
    def update_parameters(x):
        global update
        update = True

    # Create trackbars for adjusting morphological operations on the GrabCut mask.
    cv2.createTrackbar('PR_FGD Close', 'Config GrabCut', close_PR_FGD, 90, update_parameters)
    cv2.createTrackbar('PR_FGD Erode', 'Config GrabCut', erode_PR_FGD, 90, update_parameters)
    cv2.createTrackbar('FGD Erode', 'Config GrabCut', erode_FGD, 90, update_parameters)
    cv2.createTrackbar('FGD Noise', 'Config GrabCut', noise_FGD, 90, update_parameters)
    cv2.createTrackbar('BGD Erode', 'Config GrabCut', erode_BGD, 90, update_parameters)
    
    # Display the original mask.
    im_show_max(img_before, window_name='Before GrabCut', max_resolution=400)

    while True:
        if update:
            # Retrieve current positions of the trackbars.
            close_PR_FGD = make_odd(cv2.getTrackbarPos('PR_FGD Close', 'Config GrabCut'))
            erode_PR_FGD = cv2.getTrackbarPos('PR_FGD Erode', 'Config GrabCut')
            erode_FGD    = cv2.getTrackbarPos('FGD Erode', 'Config GrabCut')
            noise_FGD    = make_odd(cv2.getTrackbarPos('FGD Noise', 'Config GrabCut'))
            erode_BGD    = make_odd(cv2.getTrackbarPos('BGD Erode', 'Config GrabCut'))
            
            if noise_FGD > 0:
                # Generate the GrabCut mask based on the adjusted parameters.
                mask_GrabCut = make_mask_GrabCut(
                    mask_base, 
                    close_PR_FGD, 
                    erode_PR_FGD, 
                    erode_FGD, 
                    noise_FGD, 
                    erode_BGD
                )
                
                # Convert the GrabCut mask to a color heatmap for visualization.
                mask_color = convert_grabCut2heatmap(mask_GrabCut)

            # Display the color-mapped mask.
            im_show_max(mask_color, window_name='After GrabCut', max_resolution=800)
            
            # Keep the configuration window on top.
            cv2.setWindowProperty('Config GrabCut', cv2.WND_PROP_TOPMOST, 1)
            update = False

        # Exit the loop when 'q' is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Close all windows.
    cv2.destroyWindow("Before GrabCut")
    cv2.destroyWindow("After GrabCut")
    cv2.destroyWindow("Config GrabCut")

    return mask_GrabCut, mask_color, close_PR_FGD, erode_PR_FGD, erode_FGD, noise_FGD, erode_BGD


def tunning_HoughCircles(image, minDist=10, param1=10, param2=200, minRadius=10, maxRadius=200):
    """
    Provides an interactive tool to tune Hough Circle Transform parameters for detecting circular objects.

    **Scenario:**
    In fruit detection, accurately detecting round fruits like apples or oranges requires fine-tuning Hough Circle parameters to balance detection sensitivity and accuracy.

    **Logic and Computer Vision Methods:**
    The function creates GUI windows with trackbars for adjusting parameters like minimum distance between circles, Canny edge detector thresholds, and radius ranges. As parameters are adjusted, the Hough Circle Transform is applied in real-time, displaying detected circles on the image for immediate feedback.

    Args:
        image (numpy.ndarray): The input image to be processed.
        minDist (int, optional): Minimum distance between the centers of detected circles. Default is 10.
        param1 (int, optional): Higher threshold for the Canny edge detector. Default is 10.
        param2 (int, optional): Accumulator threshold for the circle centers. Default is 200.
        minRadius (int, optional): Minimum radius of circles to detect. Default is 10.
        maxRadius (int, optional): Maximum radius of circles to detect. If 0, it is set to twice the minimum radius. Default is 200.

    Returns:
        tuple: A tuple containing the image with detected circles drawn and the final parameter values.
    """
    # Resize the image for easier viewing and processing.
    image = resize_image(image, max_resolution=800)
    img_before = image.copy()
    # Convert the image to grayscale for circle detection.
    grayFrame = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Apply Gaussian blur to reduce noise and improve circle detection.
    blurFrame = cv2.GaussianBlur(grayFrame, (3, 3), 0)

    # Define window names for before and after images, and for configuration.
    window_before = 'Before Circles'
    window_after = 'After Circles'
    window_config = 'Config Circles'

    # Create GUI windows.
    cv2.destroyAllWindows()
    cv2.namedWindow(window_before)
    cv2.moveWindow(window_before, 0, 0)

    cv2.namedWindow(window_after)
    cv2.moveWindow(window_after, 0, 525)

    cv2.namedWindow(window_config)
    cv2.moveWindow(window_config, 0, 1200)
    cv2.resizeWindow(window_config, 700, 210)

    global update
    update = False

    # Callback function for trackbar events.
    def update_parameters(x):
        global update
        update = True

    # Create trackbars for adjusting Hough Circle parameters.
    cv2.createTrackbar('minDist', 'Config Circles', minDist, 200, update_parameters)
    cv2.createTrackbar('param1', 'Config Circles', param1, 200, update_parameters)
    cv2.createTrackbar('param2', 'Config Circles', param2, 200, update_parameters)
    cv2.createTrackbar('minRadius', 'Config Circles', minRadius, 200, update_parameters)
    cv2.createTrackbar('maxRadius', 'Config Circles', maxRadius, 200, update_parameters)

    # Display the original image.
    im_show_max(img_before, window_name='Before Circles', max_resolution=600)

    while True:
        if update:
            # Retrieve current positions of the trackbars.
            minDist = cv2.getTrackbarPos('minDist', 'Config Circles')
            param1 = cv2.getTrackbarPos('param1', 'Config Circles')
            param2 = cv2.getTrackbarPos('param2', 'Config Circles')
            minRadius = cv2.getTrackbarPos('minRadius', 'Config Circles')                
            maxRadius = cv2.getTrackbarPos('maxRadius', 'Config Circles')                

            img_tmp = img_before.copy()

            # Apply Hough Circle Transform with the current parameters.
            if minDist > 1:
                circles = cv2.HoughCircles(
                    blurFrame, 
                    cv2.HOUGH_GRADIENT, 
                    dp=1.0, 
                    minDist=minDist, 
                    param1=param1, 
                    param2=param2, 
                    minRadius=minRadius, 
                    maxRadius=maxRadius
                )
        
            if circles is not None: 
                # Round the circle parameters and convert to integer.
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]: 
                    # Draw the center of the circle.
                    cv2.circle(img_tmp, (i[0], i[1]), 1, (0, 100, 100), 3)
                    # Draw the circumference of the circle.
                    cv2.circle(img_tmp, (i[0], i[1]), i[2], (255, 0, 0), 3)

            # Display the image with detected circles.
            im_show_max(img_tmp, window_name='After Circles', max_resolution=800)
            # Keep the configuration window on top.
            cv2.setWindowProperty('Config Circles', cv2.WND_PROP_TOPMOST, 1)
            update = False

        # Exit the loop when 'q' is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Close all windows.
    cv2.destroyWindow("Before Circles")
    cv2.destroyWindow("After Circles")
    cv2.destroyWindow("Config Circles")

    return img_tmp, minDist, param1, param2, minRadius, maxRadius


def remove_saturation_from_background(image, mask_background, default_color=None):
    """
    Reduces the saturation of background areas in an image based on a provided mask.

    **Scenario:**
    In fruit detection, desaturating the background can help in emphasizing the colorful fruits, making them stand out more prominently against a muted background.

    **Logic and Computer Vision Methods:**
    The function converts the image to HSV color space, applies Gaussian blur to smooth the background, desaturates the background regions, and darkens them to reduce their visual prominence. The foreground remains untouched, enhancing the contrast between fruits and the background.

    Args:
        image (numpy.ndarray): The input image in RGB color space.
        mask_background (numpy.ndarray): A binary mask where the background regions are white (255) and foreground is black (0).
        default_color (int, optional): If provided, sets the hue of the background to a specific value. Default is None.

    Returns:
        numpy.ndarray: The image with reduced saturation in background areas.
    """
    # Make a copy of the image and convert it to HSV color space.
    image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Resize the mask to match the image dimensions if necessary.
    if mask_background.shape[:2] != image.shape[:2]:
        mask_background = cv2.resize(mask_background, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Create a copy of the HSV image for background processing.
    img_background = image.copy()
    # Apply Gaussian blur to smooth the background regions.
    img_background = blur_image(img_background, kernel_size=11)
    if default_color:
        # If a default color is specified, set the hue of the background.
        img_background[..., 0] = default_color / 2
    # Desaturate the background by setting the saturation channel to zero.
    img_background[..., 1] = img_background[..., 1] * 0.0
    # Darken the background by reducing the value channel.
    img_background[..., 2] = img_background[..., 2] * 0.5
    # Apply the background mask to the desaturated and darkened background.
    img_background = cv2.bitwise_and(img_background, img_background, mask=mask_background)

    # Extract the foreground by inverting the background mask.
    img_foreground = cv2.bitwise_and(image, image, mask=~mask_background)
    # Combine the foreground and processed background.
    img_after = cv2.bitwise_or(img_foreground, img_background)
    # Convert the image back to RGB color space.
    img_after = cv2.cvtColor(img_after, cv2.COLOR_HSV2RGB)
    return img_after


def tunning_texture(image, kernel_size=21, threshold_value=15, noise=3, expand=3, it=1):
    """
    Provides an interactive tool to tune texture detection parameters using Gaussian blur and thresholding.

    **Scenario:**
    When detecting fruits, identifying smooth areas (like backgrounds) can help in focusing detection algorithms on textured fruit surfaces.

    **Logic and Computer Vision Methods:**
    The function creates GUI windows with trackbars for adjusting Gaussian blur size, threshold values, noise removal, and mask expansion iterations. As parameters are adjusted, the image is processed in real-time to detect smooth areas, allowing for precise tuning of texture-based segmentation.

    Args:
        image (numpy.ndarray): The input image in RGB color space.
        kernel_size (int, optional): The kernel size for the Gaussian blur. Must be odd. Default is 21.
        threshold_value (int, optional): The threshold value for detecting smooth areas. Default is 15.
        noise (int, optional): The kernel size for salt-and-pepper noise removal. Default is 3.
        expand (int, optional): The kernel size for mask dilation (expansion). Default is 3.
        it (int, optional): The number of iterations for mask expansion. Default is 1.

    Returns:
        tuple: A tuple containing the processed image, the mask, and the final parameter values.
    """
    # Resize the image for easier processing.
    img_before = resize_image(image, max_resolution=800)

    # Define window names for before and after images, and for configuration.
    window_before = 'Before Texture'
    window_after = 'After Texture'
    window_config = 'Config Texture'

    # Create GUI windows.
    cv2.destroyAllWindows()
    cv2.namedWindow(window_before)
    cv2.moveWindow(window_before, 0, 0)
    cv2.namedWindow(window_after)
    cv2.moveWindow(window_after, 0, 525)
    cv2.namedWindow(window_config)
    cv2.moveWindow(window_config, 0, 1200)
    cv2.resizeWindow(window_config, 700, 210)

    global update
    update = False

    # Callback function for trackbar events.
    def update_parameters(x):
        global update
        update = True

    # Create trackbars for adjusting texture detection parameters.
    cv2.createTrackbar('size', 'Config Texture', kernel_size, 250, update_parameters)
    cv2.createTrackbar('threshold', 'Config Texture', threshold_value, 255, update_parameters)
    cv2.createTrackbar('noise', 'Config Texture', noise, 29, update_parameters)
    cv2.createTrackbar('expand', 'Config Texture', expand, 29, update_parameters)
    cv2.createTrackbar('it', 'Config Texture', it, 29, update_parameters)

    while True:
        if update:
            # Retrieve current positions of the trackbars.
            threshold_value = cv2.getTrackbarPos('threshold', 'Config Texture')
            kernel_size = make_odd(max(1, cv2.getTrackbarPos('size', 'Config Texture')))
            noise = make_odd(max(1, cv2.getTrackbarPos('noise', 'Config Texture')))
            expand = make_odd(cv2.getTrackbarPos('expand', 'Config Texture'))
            it = cv2.getTrackbarPos('it', 'Config Texture')

            img_tmp = img_before.copy()
            if kernel_size > 0:
                # Detect smooth areas based on the current parameters.
                img_after, mask_tmp = detect_smooth_areas_rgb(
                    img_tmp, 
                    kernel_size=kernel_size, 
                    threshold_value=threshold_value,
                    noise=noise,
                    expand=expand,
                    it=it
                )
                
            # Display the original and mask images side by side.
            img_window_before = build_mosaic([img_before, mask_tmp], mosaic_dims=(1, 2))
            im_show_max(img_window_before, window_name='Before Texture', max_resolution=800)
            # Display the image with smooth areas highlighted.
            im_show_max(img_after, window_name='After Texture', max_resolution=800)

            # Keep the configuration window on top.
            cv2.setWindowProperty('Config Texture', cv2.WND_PROP_TOPMOST, 1)
            update = False

        # Exit the loop when 'q' is pressed.
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Close all windows.
    cv2.destroyWindow("Before Texture")
    cv2.destroyWindow("After Texture")
    cv2.destroyWindow("Config Texture")

    return img_after, mask_tmp, kernel_size, threshold_value, noise, expand, it


def show_mosaic(images, headers=[""], footers=[""], window_name="Mosaic", mosaic_dims=(0, 0), max_resolution=1200):
    """
    Displays a mosaic of multiple images with optional headers and footers.

    **Scenario:**
    In fruit detection, visualizing different stages of image processing (e.g., original, masked, segmented) side by side helps in assessing the effectiveness of each processing step.

    **Logic and Computer Vision Methods:**
    The function utilizes the build_mosaic function to arrange images in a grid layout, optionally adding headers and footers for each column. It resizes images to fit within a specified maximum resolution and ensures that all images are in color format before creating the mosaic.

    Args:
        images (list or numpy.ndarray): A list of images to include in the mosaic.
        headers (list of str, optional): Optional headers for each column in the mosaic. Default is [""].
        footers (list of str, optional): Optional footers for each column in the mosaic. Default is [""].
        window_name (str, optional): The name of the window to display the mosaic. Default is "Mosaic".
        mosaic_dims (tuple, optional): The dimensions of the mosaic grid as (rows, columns). Default is (0, 0).
        max_resolution (int, optional): The maximum resolution for resizing images before creating the mosaic. Default is 1200.

    Returns:
        None
    """
    # Ensure that a window name is provided and is a string.
    assert window_name, "window_name must be provided."
    assert isinstance(window_name, str), "window_name must be a string."

    # Build the mosaic image using the provided images and parameters.
    mosaic = build_mosaic(
        images, 
        mosaic_dims=mosaic_dims, 
        headers=headers, 
        footers=footers, 
        max_resolution=max_resolution
    )
    height, width = mosaic.shape[:2]

    if NOTEBOOK_MODE:
        # Display the mosaic in a Jupyter Notebook.
        plt.figure(figsize=(10, 10))
        plt.imshow(mosaic)  # Convert from BGR to RGB for correct color display
        plt.axis('off')  # Turn off axis
        plt.title(window_name, fontsize=16, fontweight='bold')  # Add title
        plt.show()
    else:
        # Destroy any existing windows and create a new one for the mosaic.
        cv2.destroyAllWindows()
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # Keep the mosaic window on top of other windows.
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        # Move the mosaic window to the top-left corner of the screen.
        cv2.moveWindow(window_name, 0, 0)
        # Display the mosaic image.
        im_show_max(mosaic, window_name=window_name, max_resolution=max_resolution)
        # Resize the window to match the mosaic image's dimensions.
        cv2.resizeWindow(window_name, width, height)
        # Wait for a key press to ensure the window remains open.
        cv2.waitKey(WAIT_TIME)

        
    
def draw_ellipse_by_factor(image, factor=(0.5, 0.5), color=(255, 255, 255), thickness=-1):
    """
    Draws an ellipse on the image based on specified width and height factors.

    **Scenario:**
    In fruit detection, drawing ellipses can help in highlighting areas of interest or in masking out irrelevant regions.

    **Logic and Computer Vision Methods:**
    The function calculates the ellipse's size and position based on the image dimensions and the provided factors. It then uses OpenCV's ellipse drawing function to render the ellipse onto the image.

    Args:
        image (numpy.ndarray): The input image on which to draw the ellipse.
        factor (tuple, optional): The width and height factors relative to the image size. Default is (0.5, 0.5).
        color (tuple, optional): The color of the ellipse in BGR format. Default is white.
        thickness (int, optional): The thickness of the ellipse border. Use -1 for a filled ellipse. Default is -1.

    Returns:
        None
    """
    # Determine the dimensions of the ellipse based on the provided factors.
    h, w = image.shape[:2]
    height, width = int(h * factor[1]), int(w * factor[0])
    y, x = int((h - height) // 2), int((w - width) // 2)

    # Calculate the center point of the ellipse.
    center = (x + width // 2, y + height // 2)

    # Define the axes lengths for the ellipse.
    axesLength = (width // 2, height // 2)  # Half of the specified dimensions.
    angle = 0  # No rotation.
    startAngle = 0  # Start angle of the ellipse arc.
    endAngle = 360  # End angle of the ellipse arc.

    # Draw the ellipse on the image.
    cv2.ellipse(image, center, axesLength, angle, startAngle, endAngle, color, thickness)


def detect_contours(mask, min_number_of_points=0):
    """
    Detects contours in a binary image and filters them based on the minimum number of points.

    **Scenario:**
    After segmenting fruits from the background, detecting contours helps in outlining each fruit for further analysis like counting or measuring size.

    **Logic and Computer Vision Methods:**
    The function uses OpenCV's `findContours` to identify contours in the binary mask. It then filters these contours to exclude those with fewer points than the specified minimum, removing insignificant or noise-related contours.

    Args:
        mask (numpy.ndarray): The binary image in which contours will be detected.
        min_number_of_points (int, optional): The minimum number of points a contour must have to be considered valid. Default is 0.

    Returns:
        list: A list of detected contours that meet the minimum points requirement.
    """
    # Convert the mask to uint8 type for contour detection.
    mask = np.array(mask, dtype=np.uint8)
    # Find external contours using the TC89_KCOS approximation method.
    contours_list, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    debug_print(f'Found {len(contours_list)} contours.')

    # Filter contours by the number of points using list comprehension.
    if min_number_of_points > 0:
        filtered_contours = [cnt for cnt in contours_list if cnt.shape[0] >= min_number_of_points]
        debug_print(f'{len(contours_list)} contours reduced to {len(filtered_contours)}.')
        contours_list = filtered_contours

    return contours_list


def make_mask_GrabCut(mask, cfg_bgd_close_PR_FGD, cfg_bgd_erode_PR_FGD, cfg_bgd_erode_FGD, cfg_bgd_noise_FGD, cfg_bgd_erode_BGD):
    """
    Creates a GrabCut mask by applying morphological operations based on provided configurations.

    **Scenario:**
    When segmenting fruits from complex backgrounds, refining the initial mask using morphological operations ensures more accurate GrabCut results.

    **Logic and Computer Vision Methods:**
    The function applies closing and erosion to probable foreground areas, erodes definite foregrounds, removes noise, and erodes background areas. It then constructs the final GrabCut mask by assigning appropriate GrabCut mask values based on the processed regions.

    Args:
        mask (numpy.ndarray): The initial binary mask for GrabCut.
        cfg_bgd_close_PR_FGD (int): Kernel size for closing probable foreground regions.
        cfg_bgd_erode_PR_FGD (int): Kernel size for erosion on probable foreground regions.
        cfg_bgd_erode_FGD (int): Kernel size for erosion on definite foreground regions.
        cfg_bgd_noise_FGD (int): Kernel size for noise removal in definite foreground regions.
        cfg_bgd_erode_BGD (int): Kernel size for erosion on background regions.

    Returns:
        numpy.ndarray: The refined GrabCut mask.
    """
    # Create a copy of the foreground mask.
    mask_FGD = mask.copy()
    # Apply morphological closing to probable foreground regions.
    mask_PR_FGD = close_mask_circle(mask_FGD, kernel_size=cfg_bgd_close_PR_FGD, iterations=1)
    # Apply erosion to probable foreground regions.
    mask_PR_FGD = erode_mask_circle(mask_PR_FGD, kernel_size=cfg_bgd_erode_PR_FGD, iterations=1)
    if cfg_bgd_erode_FGD < 90:
        # Erode definite foreground regions to refine them.
        mask_FGD = erode_mask_circle(mask_FGD, kernel_size=cfg_bgd_erode_FGD, iterations=1)
        # Remove noise from the definite foreground.
        mask_FGD = remove_salt_and_pepper(mask_FGD, kernel_size=cfg_bgd_noise_FGD)
    else:
        # If erosion size is too large, reset the foreground mask.
        mask_FGD = np.zeros_like(mask_FGD)
    # Erode background regions to separate them from foreground.
    mask_BGD = erode_mask_circle(~mask_PR_FGD, kernel_size=cfg_bgd_erode_BGD, iterations=2)

    # Assign GrabCut mask values based on the processed regions.
    mask_GrabCut = np.where(mask_BGD == 255, cv2.GC_BGD, cv2.GC_PR_BGD).astype('uint8')
    mask_GrabCut = np.where(mask_PR_FGD == 255, cv2.GC_PR_FGD, mask_GrabCut).astype('uint8')
    mask_GrabCut = np.where(mask_FGD == 255, cv2.GC_FGD, mask_GrabCut).astype('uint8')
    return mask_GrabCut


def remove_background_GrabCut(image, mask, rescale_factor=1.0):
    """
    Applies the GrabCut algorithm to segment the foreground from the background.

    **Scenario:**
    In fruit detection, accurately separating fruits from complex backgrounds is crucial. GrabCut assists in refining the segmentation based on an initial mask.

    **Logic and Computer Vision Methods:**
    The function optionally rescales the image for faster processing, initializes background and foreground models, applies GrabCut with the provided mask, refines the mask, and applies it to the original image to extract the foreground.

    Args:
        image (numpy.ndarray): The input image on which GrabCut will be applied.
        mask (numpy.ndarray): An initial mask defining the rough areas of foreground and background.
        rescale_factor (float, optional): A scaling factor to resize the image and mask for faster processing. Must be between 0.2 and 1.0. Default is 1.0.

    Returns:
        tuple: A tuple containing the segmented image after GrabCut and the refined mask.
    """
    # Ensure the rescale factor is within the valid range.
    assert 0.2 < rescale_factor <= 1.0, "rescale_factor must be in the range (0.2, 1.0]"
    
    # Make a copy of the original image for reference.
    img_before = image.copy()

    if rescale_factor < 1.0:
        # Calculate the new size based on the rescale factor.
        rescaled_shape = (int(image.shape[1] * rescale_factor), int(image.shape[0] * rescale_factor))
        
        # Resize the image and mask for faster processing.
        image = cv2.resize(image, rescaled_shape, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, rescaled_shape, interpolation=cv2.INTER_NEAREST)  # Avoid interpolation.

    # Initialize background and foreground models required by GrabCut.
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Apply the GrabCut algorithm to the resized image and mask.
    cv2.grabCut(image, mask, None, bgdModel, fgdModel, iterCount=3, mode=cv2.GC_INIT_WITH_MASK)

    # Refine the mask: Set probable and definite background to 0, probable and definite foreground to 1.
    mask = np.where(
        (mask == cv2.GC_PR_BGD) | 
        (mask == cv2.GC_BGD), 
        0, 
        1
    ).astype('uint8')
    
    if rescale_factor < 1.0:
        # Resize the refined mask back to the original image size.
        mask = cv2.resize(mask, (img_before.shape[1], img_before.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Apply the refined mask to the original image to extract the foreground.
    img_after = img_before * mask[:, :, np.newaxis]

    return img_after, mask


def convert_grabCut2grayscale(mask_GrabCut):
    """
    Converts a GrabCut mask to a grayscale image with specific color mappings for visualization.

    **Scenario:**
    Visualizing the GrabCut mask helps in understanding the segmentation results, distinguishing between foreground and background areas.

    **Logic and Computer Vision Methods:**
    The function maps GrabCut mask values to specific grayscale intensities: black for background, white for foreground, light gray for probable foreground, and dark gray for probable background.

    Args:
        mask_GrabCut (numpy.ndarray): The GrabCut mask with values indicating background and foreground.

    Returns:
        numpy.ndarray: A grayscale image representing the GrabCut mask with distinct intensities.
    """
    # Initialize a grayscale mask with zeros.
    mask_grayscale = np.zeros_like(mask_GrabCut, dtype=np.uint8)

    # Map GrabCut mask values to grayscale intensities.
    mask_grayscale[mask_GrabCut == cv2.GC_BGD] = 0       # Black for background.
    mask_grayscale[mask_GrabCut == cv2.GC_FGD] = 255    # White for foreground.
    mask_grayscale[mask_GrabCut == cv2.GC_PR_FGD] = 170 # Light gray for probable foreground.
    mask_grayscale[mask_GrabCut == cv2.GC_PR_BGD] = 100 # Dark gray for probable background.

    return mask_grayscale


def im_show_max(image, window_name="Image", max_resolution=800):
    """
    Displays an image in a window after resizing it to a maximum resolution.

    **Scenario:**
    Quickly visualizing processed images (e.g., after segmentation or filtering) helps in assessing the effectiveness of the applied techniques.

    **Logic and Computer Vision Methods:**
    The function resizes the image to ensure it fits within the specified resolution, converts it from BGR to RGB color space for correct color representation, and displays it using OpenCV's `imshow`.

    Args:
        image (numpy.ndarray): The image to be displayed.
        window_name (str, optional): The name of the display window. Default is "Image".
        max_resolution (int, optional): The maximum width or height of the displayed image. Default is 800.

    Returns:
        None
    """
    # Ensure the image has either 2 (grayscale) or 3 (color) channels.
    assert len(image.shape) == 2 or image.shape[2] == 3, "The image must be grayscale or color."
    # Ensure a valid window name is provided.
    assert window_name is not None, "The title must be a string."
    # Resize the image and convert from BGR to RGB for correct color display.
    resized_image = resize_image(image, max_resolution=max_resolution)
    if len(resized_image.shape) == 3 and resized_image.shape[2] == 3:
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    # Display the image in the specified window.
    cv2.imshow(window_name, resized_image)


def rescale_image(image, dimension, filter_range, distinct_categories, new_range):
    """
    Rescales specific channels of an image based on filtering and categorization parameters.

    **Scenario:**
    In fruit detection, adjusting specific color channels can enhance the segmentation process by emphasizing particular color ranges associated with fruits.

    **Logic and Computer Vision Methods:**
    The function converts the image to HSV color space, scales the specified dimension, filters values within a range, categorizes them into distinct groups, and assigns new values based on the categorization. This process enhances or modifies specific color channels to aid in better segmentation or detection.

    Args:
        image (numpy.ndarray): The input image in RGB color space.
        dimension (int): The channel to process (0 for Hue, 1 for Saturation, 2 for Value in HSV).
        filter_range (tuple): The range of values to filter within the specified dimension.
        distinct_categories (int): The number of distinct categories to divide the filtered range into.
        new_range (tuple): The new range of values to assign to each category.

    Returns:
        numpy.ndarray: The image after rescaling the specified dimension.
    """
    # Convert the image from RGB to HSV color space.
    new_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.uint8)

    # Adjust the scale of color parameters based on the dimension.
    if dimension == 0:
        factor = 255.0 / 360.0
    else:
        factor = 255.0 / 100.0

    # Scale the filter and new ranges to match the HSV channel scales.
    filter_range = (np.array(filter_range) * factor).astype(np.uint8)
    new_range = (np.array(new_range) * factor).astype(np.uint8)

    new_range_size = new_range[1] - new_range[0]
    filter_range_size = filter_range[1] - filter_range[0]

    # Select the specified channel for processing.
    selected_dimension = new_image[..., dimension]

    # Filter values within the specified range and assign new categorized values.
    new_selected_dimension = np.where(
        (selected_dimension >= filter_range[0]) & (selected_dimension <= filter_range[1]), 
        new_range[0] + 
        (selected_dimension - filter_range[0]) / filter_range_size * new_range_size // distinct_categories * distinct_categories,
        selected_dimension
    ).astype(np.uint8)
            
    # Assign the new values back to the specified channel.
    new_image[..., dimension] = new_selected_dimension

    # Convert the image back to RGB color space.
    new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)

    return new_image


def tunning_BlobCircles(image, img_original, minCircularity=1, minConvexity=1, minInertiaRatio=1, minArea=90, maxArea=1200, min_radius=10, tolerance_overlap=0.15):
    """
    Provides an interactive tool to tune blob detection parameters for identifying circular objects in images.

    **Scenario:**
    In fruit detection systems, accurately identifying and distinguishing between different types of fruits (e.g., apples, oranges) requires precise detection of their circular shapes. This function allows users to interactively adjust parameters to optimize blob detection for varying fruit sizes, shapes, and overlapping conditions.

    **Logic and Computer Vision Methods:**
    The function sets up a graphical user interface (GUI) with trackbars to adjust parameters related to blob characteristics such as circularity, convexity, inertia ratio, area, radius, and overlap tolerance. As parameters are adjusted, the blob detection algorithm processes the image in real-time, displaying the detected circles and corresponding masks. This interactive tuning helps in fine-tuning the detection algorithm to achieve optimal results under different imaging conditions.

    Args:
        image (numpy.ndarray): The input image where blobs (fruits) are to be detected.
        img_original (numpy.ndarray): A copy of the original image for reference during detection.
        minCircularity (float, optional): The minimum circularity of detected blobs. Range: 0-1. Default is 1.
        minConvexity (float, optional): The minimum convexity of detected blobs. Range: 0-1. Default is 1.
        minInertiaRatio (float, optional): The minimum inertia ratio of detected blobs. Range: 0-1. Default is 1.
        minArea (int, optional): The minimum area of detected blobs in pixels. Default is 90.
        maxArea (int, optional): The maximum area of detected blobs in pixels. Default is 1200.
        min_radius (int, optional): The minimum radius of detected circles in pixels. Default is 10.
        tolerance_overlap (float, optional): The maximum allowed overlap between detected circles. Range: 0-1. Default is 0.15.

    Returns:
        tuple: A tuple containing detected circles, the image with circles drawn, the mask of circles, and the final parameter values.
    """
    # Create copies of the input images to preserve the originals.
    img_before = image.copy()
    img_original_before = img_original.copy()
    
    # Determine if the image is in a vertical orientation to set appropriate window sizes.
    vertical = img_before.shape[0] > img_before.shape[1]
    if vertical:
        window_size = 150
    else:
        window_size = 200

    # Initialize GUI windows for displaying images and configuring parameters.
    cv2.destroyAllWindows()
    cv2.namedWindow('Before Circles')
    cv2.moveWindow('Before Circles', 0, 0)

    cv2.namedWindow('After Circles')
    cv2.moveWindow('After Circles', 0, 525)

    cv2.namedWindow('Config Circles')
    cv2.moveWindow('Config Circles', 0, 1200)
    cv2.resizeWindow('Config Circles', 700, 280)

    global update
    update = False

    # Callback function that sets the update flag when trackbar positions change.
    def update_parameters(x):
        global update
        update = True

    # Create trackbars for adjusting blob detection parameters.
    cv2.createTrackbar('Circul.', 'Config Circles', int(minCircularity*20), 20, update_parameters)
    cv2.createTrackbar('Convex.', 'Config Circles', int(minConvexity*20), 20, update_parameters)
    cv2.createTrackbar('Inertia', 'Config Circles', int(minInertiaRatio*20), 20, update_parameters)
    cv2.createTrackbar('min Area', 'Config Circles', int(minArea/10), 500, update_parameters)
    cv2.createTrackbar('max Area', 'Config Circles', int(maxArea/10), 6000, update_parameters)
    cv2.createTrackbar('Radius', 'Config Circles', min_radius, 200, update_parameters)
    cv2.createTrackbar('Overlap', 'Config Circles', int(tolerance_overlap*20), 20, update_parameters)

    # Display the initial image in the 'Before Circles' window.
    im_show_max(img_before, window_name='Before Circles', max_resolution=window_size*3)

    while True:
        if update:
            # Retrieve and adjust parameters based on trackbar positions.
            minCircularity = cv2.getTrackbarPos('Circul.', 'Config Circles') / 20.0
            minConvexity = cv2.getTrackbarPos('Convex.', 'Config Circles') / 20.0
            minInertiaRatio = cv2.getTrackbarPos('Inertia', 'Config Circles') / 20.0
            minArea = int(cv2.getTrackbarPos('min Area', 'Config Circles') * 10.0)
            maxArea = int(cv2.getTrackbarPos('max Area', 'Config Circles') * 10.0)          
            min_radius = cv2.getTrackbarPos('Radius', 'Config Circles')                
            tolerance_overlap = cv2.getTrackbarPos('Overlap', 'Config Circles') / 20.0

            # Create temporary copies of the images for processing.
            img_tmp = img_before.copy()
            img_original = img_original_before.copy()

            # Apply blob detection only if all parameters are set to valid values.
            if minCircularity > 0 and minConvexity > 0 and minInertiaRatio > 0 and minArea > 0 and maxArea > minArea: 
                circles, img_circles, mask_circles = detect_circles(
                    img_tmp, 
                    img_original,
                    minCircularity=minCircularity, 
                    minConvexity=minConvexity, 
                    minInertiaRatio=minInertiaRatio, 
                    minArea=minArea, 
                    maxArea=maxArea,
                    min_radius=min_radius,
                    tolerance_overlap=tolerance_overlap
                )

            # Display the mask and the image with detected circles.
            im_show_max(mask_circles, window_name='Before Circles', max_resolution=window_size*3)
            im_show_max(img_circles, window_name='After Circles', max_resolution=window_size*4)
            # Keep the 'Config Circles' window on top.
            cv2.setWindowProperty('Config Circles', cv2.WND_PROP_TOPMOST, 1)
            update = False

        # Exit the loop when the 'q' key is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Close all GUI windows after exiting the loop.
    cv2.destroyWindow("Before Circles")
    cv2.destroyWindow("After Circles")
    cv2.destroyWindow("Config Circles")

    # Return the detected circles, the image with circles drawn, the mask, and the final parameter values.
    return circles, img_circles, mask_circles, minCircularity, minConvexity, minInertiaRatio, minArea, maxArea, min_radius, tolerance_overlap


def rescale_image(image, dimension, filter_range, distinct_categories, new_range):
    """
    Rescales specific channels of an image based on filtering and categorization parameters.

    **Scenario:**
    In fruit detection, adjusting specific color channels can enhance the segmentation process by emphasizing particular color ranges associated with fruits. For example, increasing the prominence of red hues can help in better detection of apples.

    **Logic and Computer Vision Methods:**
    The function converts the image to HSV color space, scales the specified dimension, filters values within a range, categorizes them into distinct groups, and assigns new values based on the categorization. This process enhances or modifies specific color channels to aid in better segmentation or detection.

    Args:
        image (numpy.ndarray): The input image in RGB color space.
        dimension (int): The channel to process (0 for Hue, 1 for Saturation, 2 for Value in HSV).
        filter_range (tuple): The range of values to filter within the specified dimension.
        distinct_categories (int): The number of distinct categories to divide the filtered range into.
        new_range (tuple): The new range of values to assign to each category.

    Returns:
        numpy.ndarray: The image after rescaling the specified dimension.
    """
    # Convert the image from RGB to HSV color space for easier color manipulation.
    new_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.uint8)

    # Adjust the scale of color parameters based on the specified dimension.
    if dimension == 0:
        # Hue channel ranges from 0 to 360 degrees; scale it to 0-255.
        factor = 255.0 / 360.0
    else:
        # Saturation and Value channels range from 0 to 100; scale them to 0-255.
        factor = 255.0 / 100.0

    # Scale the filter and new ranges to match the HSV channel scales.
    filter_range = (np.array(filter_range) * factor).astype(np.uint8)
    new_range = (np.array(new_range) * factor).astype(np.uint8)

    # Calculate the size of the new range and filter range.
    new_range_size = new_range[1] - new_range[0]
    filter_range_size = filter_range[1] - filter_range[0]

    # Select the specified channel for processing.
    selected_dimension = new_image[..., dimension]

    # Apply filtering: replace values within the filter range with new categorized values.
    new_selected_dimension = np.where(
        (selected_dimension >= filter_range[0]) & (selected_dimension <= filter_range[1]), 
        new_range[0] + 
        (selected_dimension - filter_range[0]) / filter_range_size * new_range_size // distinct_categories * distinct_categories,
        selected_dimension
    ).astype(np.uint8)
            
    # Assign the new categorized values back to the specified channel.
    new_image[..., dimension] = new_selected_dimension

    # Convert the image back to RGB color space after modifications.
    new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)

    return new_image

def sort_coordinates(coordinate, threshold=100):
    """
    Sorts coordinates according to a given threshold.

    Parameters:
    - coordinate: a tuple containing the x and y coordinates, and R.
    - threshold: an integer value representing the division threshold for the y-coordinate (default is 100).

    Returns:
    - The y-coordinate value multiplied by 100000 and added to the x-coordinate.
    """
    # Unpack the coordinate tuple. Assuming 'coordinate' is structured as ((x, y), R).
    (x, y), _ = coordinate
    
    # Adjust the y-coordinate by dividing it by the threshold and scaling.
    y = (y // threshold) * 100000
    
    # Combine the adjusted y-coordinate with the x-coordinate to create a single sortable value.
    return y + x


def convert_grabCut2heatmap(mask_GrabCut):
    """
    Converts a bidimensional GrabCut mask into a 3D heatmap using predefined colors.

    Parameters:
        mask_GrabCut (np.ndarray): A 2D numpy array containing GrabCut mask values. This mask
        categorizes each pixel as background, foreground, probable background, or probable foreground
        using integers 0 through 3, respectively.

    The function maps these classifications to specific colors:
    - Background (cv2.GC_BGD, value 0): Black
    - Foreground (cv2.GC_FGD, value 1): White
    - Probable Foreground (cv2.GC_PR_FGD, value 2): Light Gray
    - Probable Background (cv2.GC_PR_BGD, value 3): Dark Gray

    This mapping is used to convert the bidimensional mask into a 3D heatmap, where each
    pixel's color corresponds to its classification.

    Returns:
        np.ndarray: A 3-dimensional numpy array of shape (height, width, 3) representing the heatmap.
        Each element of this array is a color (B, G, R) tuple corresponding to the classification
        of each pixel in the input `mask_GrabCut`.

    Note: The function uses predefined colors for the classifications and relies on `cv2`'s
    constants for the mapping. Ensure `cv2` and `numpy` are properly imported.
    """
    # Initialize a list to hold color mappings for each GrabCut classification.
    color_grab_cut = []
    
    # Insert color for Background (cv2.GC_BGD, value 0): Black
    color_grab_cut.insert(cv2.GC_BGD, (0, 0, 0))  # Black
    
    # Insert color for Foreground (cv2.GC_FGD, value 1): White
    color_grab_cut.insert(cv2.GC_FGD, (255, 255, 255))  # White
    
    # Insert color for Probable Foreground (cv2.GC_PR_FGD, value 2): Light Gray
    color_grab_cut.insert(cv2.GC_PR_FGD, (200, 200, 200))  # Light Gray
    
    # Insert color for Probable Background (cv2.GC_PR_BGD, value 3): Dark Gray
    color_grab_cut.insert(cv2.GC_PR_BGD, (100, 100, 100))  # Dark Grayq
    
    # Convert the list of colors to a NumPy array for efficient indexing.
    color_grab_cut = np.array(color_grab_cut, dtype=np.uint8)
    
    # Create the heatmap by mapping each pixel in the GrabCut mask to its corresponding color.
    heat_map = np.take(color_grab_cut, mask_GrabCut, axis=0)

    return heat_map
