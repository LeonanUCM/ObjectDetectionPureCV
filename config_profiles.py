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
            cfg_text = ""
            cfg_text += f"\n\n# Profile: {cfg.profile}"
            cfg_text += f"\n# Quality:"
            cfg_text += f"\ncfg.quantization_n_colors, cfg.max_resolution, cfg.smooth_colors, cfg.factor_contrast = {cfg.quantization_n_colors, cfg.max_resolution, cfg.smooth_colors, cfg.factor_contrast}"

            cfg_text += f"\n\n# Blur:"
            cfg_text += f"\ncfg.blur_clahe_grid, cfg.blur_clahe_limit, cfg.blur_salt_pepper, cfg.blur_size = {cfg.blur_clahe_grid, cfg.blur_clahe_limit, cfg.blur_salt_pepper, cfg.blur_size}"

            cfg_text += f"\n\n# Amplify Saturation:"
            cfg_text += f"\ncfg.color_amplify_hue, cfg.color_amplify_range, cfg.color_amplify_increase = {cfg.color_amplify_hue, cfg.color_amplify_range, cfg.color_amplify_increase}"

            cfg_text += f"\n\n# Foreground selection:"
            for i in cfg.foreground_list:
                cfg_text += f"\ncfg.foreground_list.append({i})"

            cfg_text += f"\n\n# Texture Removal:"
            cfg_text += f"\n cfg.texture_kernel_size, cfg.texture_threshold_value, cfg.texture_noise, cfg.texture_expand, cfg.texture_it = { cfg.texture_kernel_size, cfg.texture_threshold_value, cfg.texture_noise, cfg.texture_expand, cfg.texture_it}"

            cfg_text += f"\n\n# Object Selection"

            for i in cfg.mask_list:
                cfg_text += f"\ncfg.mask_list.append({i})"

            cfg_text += f"\n\n# Circle Detection:"
            cfg_text += f"\ncfg.circle_minCircularity, cfg.circle_minConvexity, cfg.circle_minInertiaRatio, = {cfg.circle_minCircularity, cfg.circle_minConvexity, cfg.circle_minInertiaRatio}"
            cfg_text += f"\ncfg.circle_minArea, cfg.circle_maxArea =  {cfg.circle_minArea, cfg.circle_maxArea}"
            cfg_text += f"\ncfg.min_radius_circle, cfg.tolerance_overlap =  {cfg.min_radius_circle, cfg.tolerance_overlap}"
            
            return cfg_text

    cfg = Config()

    cfg.profile = profile

    assert profile in ['ORANGE', 'APPLE', 'YELLOW_PEACH'], \
    "Profile must be one of: 'ORANGE', 'APPLE', 'YELLOW_PEACH'"

    # Default configuration valid for all profiles, 
    # but can be redefined in the profile specific configuration
    cfg.max_resolution = 1536 # resolution of reduced image to be processed
    cfg.mask_list = []
    cfg.foreground_list = []
    cfg.mask_list_name_weight = []
    cfg.texture_kernel_size, cfg.texture_threshold_value, cfg.texture_noise, cfg.texture_expand, cfg.texture_it = (1, 1, 1, 1, 3)


    # Load configuration depending on the cfg.profile
    if (cfg.profile == 'ORANGE'):
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
        cfg.foreground_list.append([['Foreground Probably', 1], [5, 39, 29, 61, 100, 100, 43, 9, 5, 1]])
        cfg.foreground_list.append([['Foreground Certainly', 2], [18, 60, 60, 49, 100, 100, 47, 3, 1, 1]])

        # Texture Removal:
        cfg.texture_kernel_size, cfg.texture_threshold_value, cfg.texture_noise, cfg.texture_expand, cfg.texture_it = (29, 23, 1, 7, 1)
        cfg.texture_2_kernel_size, cfg.texture_2_threshold_value, cfg.texture_2_noise, cfg.texture_2_expand, cfg.texture_2_it = (3, 1, 1, 3, 1)
        cfg.texture_3_kernel_size, cfg.texture_3_threshold_value, cfg.texture_3_noise, cfg.texture_3_expand, cfg.texture_3_it = (1, 0, 1, 1, 0)
        cfg.texture_4_kernel_size, cfg.texture_4_threshold_value, cfg.texture_4_noise, cfg.texture_4_expand, cfg.texture_4_it = (1, 0, 1, 1, 0)

        # Object Selection
        cfg.mask_list.append([['Object Low Intensity and Sat', -1], [40, 0, 0, 23, 40, 80, 7, 11, 1, 1]])
        cfg.mask_list.append([['Object Probably', 1], [16, 30, 35, 65, 100, 100, 25, 3, 3, 1]])
        cfg.mask_list.append([['Object Certainly', 2], [16, 60, 70, 65, 100, 100, 33, 3, 3, 1]])

        # Circle Detection:
        cfg.circle_minCircularity, cfg.circle_minConvexity, cfg.circle_minInertiaRatio, = (0.4, 0.4, 0.5)
        cfg.circle_minArea, cfg.circle_maxArea =  (710, 12150)
        cfg.min_radius_circle, cfg.tolerance_overlap =  (34, 0.3)

    elif (cfg.profile == 'APPLE'):
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
        cfg.foreground_list.append([['Foreground Probably', 1], [50, 30, 40, 89, 100, 100, 31, 21, 7, 1]])
        cfg.foreground_list.append([['Foreground Certainly', 2], [61, 45, 50, 76, 100, 100, 41, 43, 7, 1]])

        # Texture Removal:
        cfg.texture_kernel_size, cfg.texture_threshold_value, cfg.texture_noise, cfg.texture_expand, cfg.texture_it = (57, 13, 1, 1, 1)   
        cfg.texture_2_kernel_size, cfg.texture_2_threshold_value, cfg.texture_2_noise, cfg.texture_2_expand, cfg.texture_2_it = (19, 11, 1, 1, 1)   
        cfg.texture_3_kernel_size, cfg.texture_3_threshold_value, cfg.texture_3_noise, cfg.texture_3_expand, cfg.texture_3_it = (33, 61, 1, 1, 1)   
        cfg.texture_4_kernel_size, cfg.texture_4_threshold_value, cfg.texture_4_noise, cfg.texture_4_expand, cfg.texture_4_it = (23, 15, 3, 1, 1)   

        # Object Selection
        cfg.mask_list.append([['Object Low Intensity and Sat', -1], [0, 0, 0, 359, 50, 66, 1, 3, 1, 2]])
        cfg.mask_list.append([['Object Probably', 1], [41, 29, 36, 78, 100, 100, 39, 17, 1, 1]])
        cfg.mask_list.append([['Object Certainly', 2], [56, 40, 75, 70, 100, 100, 27, 47, 7, 1]])
        cfg.smooth_mask_certain = 15

        # Circle Detection:
        cfg.circle_minCircularity, cfg.circle_minConvexity, cfg.circle_minInertiaRatio, = (0.4, 0.4, 0.5)
        cfg.circle_minArea, cfg.circle_maxArea =  (920, 12150)
        cfg.min_radius_circle, cfg.tolerance_overlap =  (34, 0.2)

    elif (cfg.profile == 'YELLOW_PEACH'):
        # Quality:
        cfg.quantization_n_colors, cfg.max_resolution, cfg.smooth_colors, cfg.factor_contrast = (3, 1536, 19, 0.98)

        # Blur:
        cfg.blur_clahe_grid, cfg.blur_clahe_limit, cfg.blur_salt_pepper, cfg.blur_size = (5, 2.5, 5, 5)

        # Amplify Saturation:
        cfg.color_amplify_hue, cfg.color_amplify_range, cfg.color_amplify_increase = (40, 20, 0.5)
                
        # Foreground selection:
        cfg.foreground_list = []
        cfg.foreground_list.append([['Background Green', -1], [70, 30, 20, 316, 100, 100, 3, 21, 11, 1]])
        cfg.foreground_list.append([['Background Brown', -1], [0, 3, 8, 32, 39, 85, 1, 21, 11, 1]])
        cfg.foreground_list.append([['Foreground Probably', 1], [33, 40, 37, 60, 100, 100, 27, 13, -31, 1]])
        cfg.foreground_list.append([['Foreground Certainly', 2], [35, 60, 60, 50, 100, 100, 37, -7, -27, 3]])
        
        # Texture Removal:
        cfg.texture_kernel_size, cfg.texture_threshold_value, cfg.texture_noise, cfg.texture_expand, cfg.texture_it = (29, 5, 1, 3, 1)    

        # Object Selection
        cfg.mask_list.append([['Border', -1], [50, 0, 0, 100, 70, 100, 1, 1, 5, 3]])
        cfg.mask_list.append([['Object Probably', 1], [16, 30, 35, 65, 100, 100, 25, 0, -25, 6]])
        cfg.mask_list.append([['Object Certainly', 2], [16, 40, 35, 65, 100, 100, 25, -7, -35, 5]])
        cfg.mask_list.append([['Object Certainly Middle', 3], [16, 40, 35, 65, 100, 100, 25, -7, -35, 7]])

        # Circle Detection:
        cfg.circle_minCircularity, cfg.circle_minConvexity, cfg.circle_minInertiaRatio, = (0.4, 0.4, 0.5)
        cfg.circle_minArea, cfg.circle_maxArea =  (700, 12000)
        cfg.min_radius_circle, cfg.tolerance_overlap =  (23, 0.1)
      
    else:
        raise Exception('Profile not found')    
    
    return cfg
