NOTEBOOK_MODE = True
DEBUG_LEVEL = 3
WAIT_TIME = 1
BATCH_MODE = True
PRINT_REPORT_ON_IMAGE = True
VERSION='2024/dec/05'

def set_debug_level(level):
    DEBUG_LEVEL = level
    
def set_notebook_mode(mode=True):
    NOTEBOOK_MODE = mode
    
def set_batch_mode(mode=True):
    BATCH_MODE = mode

def set_print_report_on_image(mode=True):
    PRINT_REPORT_ON_IMAGE = mode


def load_config(profile):
    class Config:
        def as_text(cfg):
            print(f"\n# Profile:", cfg.profile)
            print(f"# Quality:")
            print(f"cfg.quantization_n_colors, cfg.max_resolution, cfg.smooth_colors, cfg.factor_contrast = {cfg.quantization_n_colors, cfg.max_resolution, cfg.smooth_colors, cfg.factor_contrast}")

            print(f"\n# Blur:")
            print(f"cfg.blur_clahe_grid, cfg.blur_clahe_limit, cfg.blur_salt_pepper, cfg.blur_size = {cfg.blur_clahe_grid, cfg.blur_clahe_limit, cfg.blur_salt_pepper, cfg.blur_size}")

            print(f"\n# Amplify Saturation:")
            print(f"cfg.color_amplify_hue, cfg.color_amplify_range, cfg.color_amplify_increase = {cfg.color_amplify_hue, cfg.color_amplify_range, cfg.color_amplify_increase}")

            print(f"\n# Foreground selection:")
            for i in cfg.foreground_list:
                print(f"cfg.foreground_list.append({i})")

                print(f"\n# Texture Removal:")
                try:
                    print(f"cfg.texture_1_kernel_size, cfg.texture_1_threshold_value, cfg.texture_1_noise, cfg.texture_1_expand, cfg.texture_1_it = {cfg.texture_1_kernel_size, cfg.texture_1_threshold_value, cfg.texture_1_noise, cfg.texture_1_expand, cfg.texture_1_it}")
                    print(f"cfg.texture_2_kernel_size, cfg.texture_2_threshold_value, cfg.texture_2_noise, cfg.texture_2_expand, cfg.texture_2_it = {cfg.texture_2_kernel_size, cfg.texture_2_threshold_value, cfg.texture_2_noise, cfg.texture_2_expand, cfg.texture_2_it}")
                    print(f"cfg.texture_3_kernel_size, cfg.texture_3_threshold_value, cfg.texture_3_noise, cfg.texture_3_expand, cfg.texture_3_it = {cfg.texture_3_kernel_size, cfg.texture_3_threshold_value, cfg.texture_3_noise, cfg.texture_3_expand, cfg.texture_3_it}")
                    print(f"cfg.texture_4_kernel_size, cfg.texture_4_threshold_value, cfg.texture_4_noise, cfg.texture_4_expand, cfg.texture_4_it = {cfg.texture_4_kernel_size, cfg.texture_4_threshold_value, cfg.texture_4_noise, cfg.texture_4_expand, cfg.texture_4_it}")
                except:
                    pass

            print(f"\n# Object Selection")

            for i in cfg.color_list:
                print(f"cfg.color_list.append({i})")

            print(f"cfg.smooth_mask_certain =  {cfg.smooth_mask_certain}")

            print(f"\n# Circle Detection:")
            print(f"cfg.circle_minCircularity, cfg.circle_minConvexity, cfg.circle_minInertiaRatio, = {cfg.circle_minCircularity, cfg.circle_minConvexity, cfg.circle_minInertiaRatio}")
            print(f"cfg.circle_minArea, cfg.circle_maxArea =  {cfg.circle_minArea, cfg.circle_maxArea}")
            print(f"cfg.min_radius_circle, cfg.tolerance_overlap =  {cfg.min_radius_circle, cfg.tolerance_overlap}")
        pass

    cfg = Config()

    cfg.profile = profile

    assert profile in ['ORANGE', 'ORANGE', 'APPLE', 'APPLE', 'YELLOW_PEACH', 'YELLOW_PEACH', 'RED_PEACH', 'RED_PEACH'], \
    "Profile must be one of: 'ORANGE', 'ORANGE', 'APPLE', 'APPLE', 'YELLOW_PEACH', 'YELLOW_PEACH', 'RED_PEACH', 'RED_PEACH'"

    # Default configuration valid for all profiles, 
    # but can be redefined in the profile specific configuration
    cfg.max_resolution = 1536 # resolution of reduced image to be processed
    cfg.aspect_ratio = (4,3) # 4x3
    cfg.factor_contrast = 1
    cfg.resolution_returned = 2048 # resolution of original image returned
    cfg.expand_foreground = 13
    cfg.fill_holes = True
    cfg.color_list = []
    cfg.foreground_list = []
    cfg.color_list_name_weight = []
    cfg.quantization_n_colors = 10
    cfg.smooth_colors = 19
    cfg.texture_1_kernel_size, cfg.texture_1_threshold_value, cfg.texture_1_noise, cfg.texture_1_expand, cfg.texture_1_it = (1, 1, 1, 1, 3)
    cfg.texture_2_kernel_size, cfg.texture_2_threshold_value, cfg.texture_2_noise, cfg.texture_2_expand, cfg.texture_2_it = (1, 1, 1, 1, 3)
    cfg.texture_3_kernel_size, cfg.texture_3_threshold_value, cfg.texture_3_noise, cfg.texture_3_expand, cfg.texture_3_it = (1, 1, 1, 1, 3)
    cfg.texture_4_kernel_size, cfg.texture_4_threshold_value, cfg.texture_4_noise, cfg.texture_4_expand, cfg.texture_4_it = (1, 1, 1, 1, 3)
    cfg.smooth_mask_certain = 7


    # Load configuration depending on the cfg.profile
    if (cfg.profile == 'ORANGE '):
        # Profile: ORANGE 
        # Quality:
        cfg.quantization_n_colors, cfg.max_resolution, cfg.smooth_colors, cfg.factor_contrast = (0, 1536, 0, 1.05)

        # Blur:
        cfg.blur_clahe_grid, cfg.blur_clahe_limit, cfg.blur_salt_pepper, cfg.blur_size = (5, 2.1, 5, 5)

        # Amplify Saturation:
        cfg.color_amplify_hue, cfg.color_amplify_range, cfg.color_amplify_increase = (0, 0, 0)

        # Foreground selection:
        cfg.foreground_list.append([['Foreground Green', -1], [60, 20, 20, 316, 100, 100, 1, 15, 1, 1]])
        cfg.foreground_list.append([['Foreground Brown', -1], [0, 3, 8, 32, 39, 85, 1, 19, 1, 1]])
        cfg.foreground_list.append([['Foreground Probrably', 1], [5, 39, 29, 61, 100, 100, 43, 9, 5, 1]])
        cfg.foreground_list.append([['Foreground Certainly', 2], [18, 60, 60, 49, 100, 100, 47, 3, 1, 1]])

        # Texture Removal:
        cfg.texture_1_kernel_size, cfg.texture_1_threshold_value, cfg.texture_1_noise, cfg.texture_1_expand, cfg.texture_1_it = (29, 23, 1, 7, 1)
        cfg.texture_2_kernel_size, cfg.texture_2_threshold_value, cfg.texture_2_noise, cfg.texture_2_expand, cfg.texture_2_it = (3, 1, 1, 3, 1)
        cfg.texture_3_kernel_size, cfg.texture_3_threshold_value, cfg.texture_3_noise, cfg.texture_3_expand, cfg.texture_3_it = (1, 0, 1, 1, 0)
        cfg.texture_4_kernel_size, cfg.texture_4_threshold_value, cfg.texture_4_noise, cfg.texture_4_expand, cfg.texture_4_it = (1, 0, 1, 1, 0)

        # Object Selection
        cfg.color_list.append([['Object Low Intensity and Sat', -1], [40, 0, 0, 23, 40, 80, 7, 11, 1, 1]])
        cfg.color_list.append([['Object Probrably', 1], [16, 30, 35, 65, 100, 100, 25, 3, 3, 1]])
        cfg.color_list.append([['Object Certainly', 2], [16, 60, 70, 65, 100, 100, 33, 3, 3, 1]])
        cfg.smooth_mask_certain =  0

        # Circle Detection:
        cfg.circle_minCircularity, cfg.circle_minConvexity, cfg.circle_minInertiaRatio, = (0.4, 0.4, 0.5)
        cfg.circle_minArea, cfg.circle_maxArea =  (710, 12150)
        cfg.min_radius_circle, cfg.tolerance_overlap =  (34, 0.3)

    elif (cfg.profile == 'APPLE '):
        # Profile: APPLE 
        # Quality:
        cfg.quantization_n_colors, cfg.max_resolution, cfg.smooth_colors, cfg.factor_contrast = (0, 1536, 19, 0.98)

        # Blur:
        cfg.blur_clahe_grid, cfg.blur_clahe_limit, cfg.blur_salt_pepper, cfg.blur_size = (5, 0.7, 3, 5)

        # Amplify Saturation:
        cfg.color_amplify_hue, cfg.color_amplify_range, cfg.color_amplify_increase = (75, 10, 0.3)

        # Foreground selection:
        cfg.foreground_list.append([['Foreground Green', -1], [87, 20, 20, 316, 100, 100, 5, 11, 1, 1]])
        cfg.foreground_list.append([['Foreground Brown', -1], [325, 3, 5, 53, 41, 85, 3, 19, 1, 1]])
        cfg.foreground_list.append([['Foreground Probrably', 1], [50, 30, 40, 89, 100, 100, 31, 21, 7, 1]])
        cfg.foreground_list.append([['Foreground Certainly', 2], [61, 45, 50, 76, 100, 100, 41, 43, 7, 1]])

        # Texture Removal:
        cfg.texture_1_kernel_size, cfg.texture_1_threshold_value, cfg.texture_1_noise, cfg.texture_1_expand, cfg.texture_1_it = (57, 13, 1, 1, 1)   
        cfg.texture_2_kernel_size, cfg.texture_2_threshold_value, cfg.texture_2_noise, cfg.texture_2_expand, cfg.texture_2_it = (19, 11, 1, 1, 1)   
        cfg.texture_3_kernel_size, cfg.texture_3_threshold_value, cfg.texture_3_noise, cfg.texture_3_expand, cfg.texture_3_it = (33, 61, 1, 1, 1)   
        cfg.texture_4_kernel_size, cfg.texture_4_threshold_value, cfg.texture_4_noise, cfg.texture_4_expand, cfg.texture_4_it = (23, 15, 3, 1, 1)   

        # Object Selection
        cfg.color_list.append([['Object Low Intensity and Sat', -1], [0, 0, 0, 359, 50, 66, 1, 3, 1, 2]])
        cfg.color_list.append([['Object Probrably', 1], [41, 29, 36, 78, 100, 100, 39, 17, 1, 1]])
        cfg.color_list.append([['Object Certainly', 2], [56, 40, 75, 70, 100, 100, 27, 47, 7, 1]])
        cfg.smooth_mask_certain = 15

        # Circle Detection:
        cfg.circle_minCircularity, cfg.circle_minConvexity, cfg.circle_minInertiaRatio, = (0.4, 0.4, 0.5)
        cfg.circle_minArea, cfg.circle_maxArea =  (920, 12150)
        cfg.min_radius_circle, cfg.tolerance_overlap =  (34, 0.2)

    elif (cfg.profile == 'YELLOW_PEACH '):
        # Profile: YELLOW_PEACH 
        # Quality:
        cfg.quantization_n_colors, cfg.max_resolution, cfg.smooth_colors, cfg.factor_contrast = (16, 1536, 19, 0.98)

        # Blur:
        cfg.blur_clahe_grid, cfg.blur_clahe_limit, cfg.blur_salt_pepper, cfg.blur_size = (5, 2.5, 5, 5)

        # Amplify Saturation:
        cfg.color_amplify_hue, cfg.color_amplify_range, cfg.color_amplify_increase = (0, 0, 0)

        # Foreground selection:
        cfg.foreground_list.append([['Foreground Green', -1], [70, 30, 20, 316, 100, 100, 3, 15, 1, 1]])
        cfg.foreground_list.append([['Foreground Brown', -1], [0, 3, 8, 32, 39, 85, 1, 15, 1, 1]])
        cfg.foreground_list.append([['Foreground Probrably', 1], [36, 32, 37, 66, 100, 100, 27, 19, 11, 1]])
        cfg.foreground_list.append([['Foreground Certainly', 2], [27, 60, 60, 50, 100, 100, 37, 17, 5, 1]])

        # Texture Removal:
        cfg.texture_1_kernel_size, cfg.texture_1_threshold_value, cfg.texture_1_noise, cfg.texture_1_expand, cfg.texture_1_it = (29, 5, 1, 3, 1)    
        cfg.texture_2_kernel_size, cfg.texture_2_threshold_value, cfg.texture_2_noise, cfg.texture_2_expand, cfg.texture_2_it = (3, 11, 3, 3, 1)    
        cfg.texture_3_kernel_size, cfg.texture_3_threshold_value, cfg.texture_3_noise, cfg.texture_3_expand, cfg.texture_3_it = (5, 9, 1, 1, 0)     
        cfg.texture_4_kernel_size, cfg.texture_4_threshold_value, cfg.texture_4_noise, cfg.texture_4_expand, cfg.texture_4_it = (1, 0, 1, 1, 0)     

        # Object Selection
        cfg.color_list.append([['Object Low Intensity and Sat', -1], [40, 0, 0, 23, 30, 80, 5, 7, 1, 1]])
        cfg.color_list.append([['Object Probrably', 1], [16, 30, 35, 65, 100, 100, 25, 3, 3, 1]])
        cfg.color_list.append([['Object Certainly', 2], [16, 53, 35, 65, 100, 100, 25, 3, 3, 1]])
        cfg.smooth_mask_certain = 15

        # Circle Detection:
        cfg.circle_minCircularity, cfg.circle_minConvexity, cfg.circle_minInertiaRatio, = (0.4, 0.4, 0.5)
        cfg.circle_minArea, cfg.circle_maxArea =  (1110, 12380)
        cfg.min_radius_circle, cfg.tolerance_overlap =  (34, 0.1)
      
    elif (cfg.profile == 'RED_PEACH '):
        # Profile: RED_PEACH 
        # Quality:
        cfg.quantization_n_colors, cfg.max_resolution, cfg.smooth_colors, cfg.factor_contrast = (0, 1536, 19, 0.98)        

        # Blur:
        cfg.blur_clahe_grid, cfg.blur_clahe_limit, cfg.blur_salt_pepper, cfg.blur_size = (5, 2.5, 5, 3)

        # Amplify Saturation:
        cfg.color_amplify_hue, cfg.color_amplify_range, cfg.color_amplify_increase = (0, 0, 0)

        # Foreground selection:
        cfg.foreground_list.append([['Foreground Red LAB', 1], [20, 9, -25, 80, 127, 17, 11, 23, 1, 2]])
        cfg.foreground_list.append([['Foreground Yellow LAB', 1], [30, -25, 22, 85, 38, 85, 17, 19, 1, 2]])
        cfg.foreground_list.append([['Foreground Brown LAB', -1], [11, -8, 1, 100, 7, 27, 13, 7, 1, 5]])

        # Texture Removal:
        cfg.texture_1_kernel_size, cfg.texture_1_threshold_value, cfg.texture_1_noise, cfg.texture_1_expand, cfg.texture_1_it = (29, 5, 1, 3, 1)
        cfg.texture_2_kernel_size, cfg.texture_2_threshold_value, cfg.texture_2_noise, cfg.texture_2_expand, cfg.texture_2_it = (3, 11, 3, 3, 1)
        cfg.texture_3_kernel_size, cfg.texture_3_threshold_value, cfg.texture_3_noise, cfg.texture_3_expand, cfg.texture_3_it = (1, 0, 1, 1, 0)
        cfg.texture_4_kernel_size, cfg.texture_4_threshold_value, cfg.texture_4_noise, cfg.texture_4_expand, cfg.texture_4_it = (1, 0, 1, 1, 0)

        # Object Selection
        cfg.color_list.append([['Object Low Intensity and Sat', -3], [353, 0, 0, 30, 22, 75, 5, 7, 1, 1]])
        cfg.color_list.append([['Object Probrably Red', 1], [196, 20, 35, 17, 100, 100, 25, 21, 3, 1]])
        cfg.color_list.append([['Object Probrably Yellow', 1], [15, 20, 35, 70, 100, 100, 33, 21, 3, 1]])
        cfg.color_list.append([['Object Certainly Red', 2], [305, 37, 45, 20, 100, 100, 23, 17, 3, 1]])
        cfg.color_list.append([['Object Certainly Yelow', 2], [15, 53, 57, 60, 100, 100, 21, 21, 3, 1]])
        cfg.smooth_mask_certain =  15

        # Circle Detection:
        cfg.circle_minCircularity, cfg.circle_minConvexity, cfg.circle_minInertiaRatio, = (0.4, 0.4, 0.3)
        cfg.circle_minArea, cfg.circle_maxArea =  (580, 27180)
        cfg.min_radius_circle, cfg.tolerance_overlap =  (34, 0.4)

    else:
        raise Exception('Profile not found')    
    
    return cfg
