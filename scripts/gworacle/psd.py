import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
from scipy.interpolate import interp1d # Needed for interpolation

# --- Logging Setup ---
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)


# --- Axis Line Detection Function ---
def find_plot_axis_lines_simplified(img):
    """
    Detects the main horizontal (bottom) and vertical (left) axis lines
    using Hough transform on a mask of grey pixels.

    Args:
        img: The full input image (BGR, numpy array), potentially alpha-blended.

    Returns:
        A dictionary {'x_axis': (x_start, x_end, y), 'y_axis': (y_start, y_end, x)}
        defining the pixel coordinates and adjusted extent of the axes,
        or None if not found reliably.
    """
    if img is None or len(img.shape) < 2:
        logger.error("Invalid input image provided.")
        return None
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_height, img_width = img.shape[:2]
    elif len(img.shape) == 2:  # Grayscale safety check
        logger.warning("Grayscale image passed to axis detection; expected BGR.")
        return None
    else:
        logger.error(f"Unexpected image shape: {img.shape}")
        return None

    if img_height == 0 or img_width == 0:
        logger.error("Input image has zero dimension.")
        return None

    try:
        # Create closed grey mask
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Use thresholds provided in user's last code version
        lower_grey = 100
        upper_grey = 210
        grey_mask = cv2.inRange(gray_img, lower_grey, upper_grey)
        kernel_size = 3
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        closed_grey_mask = cv2.morphologyEx(grey_mask, cv2.MORPH_CLOSE, kernel)
    except Exception as e:
        logger.exception(f"Preprocessing error during axis detection: {e}")
        return None

    # Hough Line Transform on the closed grey mask for axes
    # Use parameters provided in user's last code version
    min_axis_length = min(img_width, img_height) * 0.7
    max_axis_gap = 20
    hough_threshold_axis = 30

    lines = cv2.HoughLinesP(
        closed_grey_mask,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold_axis,
        minLineLength=min_axis_length,
        maxLineGap=max_axis_gap
    )

    if lines is None:
        logger.warning(
            "HoughLinesP did not detect long lines on grey mask for axes."
        )
        return None

    horizontal_lines = []
    vertical_lines = []
    angle_tolerance_deg = 3  # Stricter tolerance for axes
    angle_tolerance_rad = np.deg2rad(angle_tolerance_deg)

    # Filter lines by orientation
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        angle = np.arctan2(y2 - y1, x2 - x1)

        is_vertical = (
            abs(angle - np.pi / 2) < angle_tolerance_rad
            or abs(angle + np.pi / 2) < angle_tolerance_rad
        )
        is_horizontal = (
            abs(angle) < angle_tolerance_rad
            or abs(abs(angle) - np.pi) < angle_tolerance_rad
        )

        if is_vertical:
            vertical_lines.append({
                'x': (x1 + x2) / 2,
                'y1': min(y1, y2),
                'y2': max(y1, y2),
                'len': length
            })
        elif is_horizontal:
            horizontal_lines.append({
                'y': (y1 + y2) / 2,
                'x1': min(x1, x2),
                'x2': max(x1, x2),
                'len': length
            })

    # Select X-axis candidate
    candidate_xaxis = [l for l in horizontal_lines if l['y'] > img_height * 0.5]
    if not candidate_xaxis:
        logger.warning("No candidate horizontal axes found in lower half.")
        return None
    candidate_xaxis.sort(key=lambda l: (-l['y'], -l['len'])) # Lowest, then longest
    best_xaxis = candidate_xaxis[0]
    orig_x_start = best_xaxis['x1']
    orig_x_end = best_xaxis['x2']
    y_axis_coord = best_xaxis['y']

    # Select Y-axis candidate
    y_axis_search_width = int(img_width * 0.25)
    candidate_yaxis = [
        l for l in vertical_lines if l['x'] < y_axis_search_width
    ]
    if not candidate_yaxis:
        logger.warning(
            f"No candidate vertical axes found in left {y_axis_search_width}px."
        )
        return None
    candidate_yaxis.sort(key=lambda l: (-l['len'], l['x'])) # Longest, then leftmost
    best_yaxis = candidate_yaxis[0]
    orig_y_start = best_yaxis['y1']
    orig_y_end = best_yaxis['y2']
    x_axis_coord = best_yaxis['x']

    # Adjust start/end points to meet at intersection
    adjusted_x_start = x_axis_coord
    adjusted_x_end = orig_x_end
    adjusted_y_start = orig_y_start
    adjusted_y_end = y_axis_coord

    # Sanity checks
    if adjusted_x_start >= adjusted_x_end or adjusted_y_start >= adjusted_y_end:
        logger.warning(
            f"Axis adjustment invalid: X=({adjusted_x_start:.1f}-{adjusted_x_end:.1f}), "
            f"Y=({adjusted_y_start:.1f}-{adjusted_y_end:.1f})"
        )
        return None

    logger.info(
        f"Adj. X-axis: y={y_axis_coord:.1f}, "
        f"x=({adjusted_x_start:.1f}-{adjusted_x_end:.1f})"
    )
    logger.info(
        f"Adj. Y-axis: x={x_axis_coord:.1f}, "
        f"y=({adjusted_y_start:.1f}-{adjusted_y_end:.1f})"
    )

    return {
        'x_axis': (adjusted_x_start, adjusted_x_end, y_axis_coord),
        'y_axis': (adjusted_y_start, adjusted_y_end, x_axis_coord)
    }


# --- Helper: Find Major Ticks (Scanline Method - Simplified Length Diff) ---
def find_major_ticks_by_scanline(
    img_bgr,
    axis_params,
    orientation='horizontal',
    scan_offset=1,
    max_scan_depth=5, # Using user's last value
    min_tick_length=2,
    consolidation_dist=3,
    major_tick_len_threshold=1.4, # Heuristic multiplier
    dark_threshold=80
):
    """
    Finds potential major tick marks extending outwards by scanning pixel lines
    on a thresholded image. Differentiates ONLY by thresholding length.

    Args:
        img_bgr: Input BGR image.
        axis_params: Dictionary with detected 'x_axis' and 'y_axis' tuples.
        orientation: 'horizontal' (scan for vertical ticks on x-axis) or 'vertical'.
        scan_offset: Pixels away from axis to start scan.
        max_scan_depth: Max pixels outwards to scan.
        min_tick_length: Minimum contiguous pixels to count as a tick.
        consolidation_dist: Max distance along axis to group detections.
        major_tick_len_threshold: Multiplier for median length in fallback heuristic.
        dark_threshold: Pixel value below which is considered 'dark' (0-255).

    Returns:
        List of detected major ticks, each as {'pos': coordinate, 'len': length}.
    """
    
    if img_bgr is None:
        logger.error("Scanline input img_bgr is None.")
        return []
    if axis_params is None or 'x_axis' not in axis_params or 'y_axis' not in axis_params:
        logger.error("Valid axis_params required for scanline tick detection.")
        return []

    img_height, img_width = img_bgr.shape[:2]
    raw_ticks = []

    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        ret, thresh_ticks = cv2.threshold(
            gray, dark_threshold, 255, cv2.THRESH_BINARY_INV
        )
    except Exception as e:
        logger.exception(f"Error thresholding ticks: {e}")
        return []

    # --- Perform Scan ---
    try:
        x_start, x_end, y_axis = map(float, axis_params['x_axis'])
        y_start, y_end, x_axis = map(float, axis_params['y_axis'])
    except (ValueError, TypeError, KeyError):
        logger.error("Invalid axis parameters received for scanline.")
        return []

    if orientation == 'horizontal': # Scan vertically DOWNWARDS from X-axis
        y_scan_start = int(round(y_axis)) + scan_offset
        x_range_start = int(round(x_start))
        x_range_end = int(round(x_end))
        for x in range(x_range_start, x_range_end + 1):
            run_length = 0
            if not (0 <= x < img_width): continue
            y_scan_end = min(y_scan_start + max_scan_depth, img_height)
            for y in range(y_scan_start, y_scan_end):
                if thresh_ticks[y, x] == 255:
                    run_length += 1
                else:
                    break
            if run_length >= min_tick_length:
                raw_ticks.append({'pos': float(x), 'len': run_length})

    elif orientation == 'vertical': # Scan horizontally LEFTWARDS from Y-axis
        x_scan_start = int(round(x_axis)) - scan_offset # Start LEFT
        y_range_start = int(round(y_start))
        y_range_end = int(round(y_end))
        for y in range(y_range_start-5, y_range_end + 5): # Scan full Y range
            run_length = 0
            if not (0 <= y < img_height): continue
            x_scan_end = max(0, x_scan_start - max_scan_depth) # Scan limit left
            for x in range(x_scan_start, x_scan_end - 2, -1): # Scan leftwards
                 if not (0 <= x < img_width): break
                 if thresh_ticks[y, x] == 255:
                     run_length += 1
                 else:
                     break
            if run_length >= min_tick_length:
                raw_ticks.append({'pos': float(y), 'len': run_length})
    else:
        logger.error(f"Invalid orientation: {orientation}")
        return []

    if not raw_ticks:
        logger.warning(f"Scanline found no raw ticks for {orientation} axis.")
        return []

    # --- Consolidate adjacent detections ---
    raw_ticks.sort(key=lambda t: t['pos'])
    consolidated_ticks = []
    if not raw_ticks: return []

    current_group = [raw_ticks[0]]
    for i in range(1, len(raw_ticks)):
        if raw_ticks[i]['pos'] - current_group[-1]['pos'] <= consolidation_dist:
            current_group.append(raw_ticks[i])
        else:
            if current_group:
                max_len = max(t['len'] for t in current_group)
                avg_pos = np.mean([t['pos'] for t in current_group])
                consolidated_ticks.append({'pos': avg_pos, 'len': max_len})
            current_group = [raw_ticks[i]]
    # Process last group
    if current_group:
        max_len = max(t['len'] for t in current_group)
        avg_pos = np.mean([t['pos'] for t in current_group])
        consolidated_ticks.append({'pos': avg_pos, 'len': max_len})

    if not consolidated_ticks:
        logger.warning(f"No ticks after consolidation {orientation}.")
        return []
    logger.info(
        f"{orientation}: Found {len(consolidated_ticks)} consolidated ticks."
    )

    # --- Differentiate Major/Minor Ticks by Length (Fallback Heuristic ONLY) ---
    lengths = np.array([t['len'] for t in consolidated_ticks])
    major_ticks = []

    if len(lengths) > 0:
        median_len = np.median(lengths)
        # Heuristic threshold based on median length
        threshold = max(
            median_len * major_tick_len_threshold, min_tick_length * 1.1
        )
        major_ticks = [t for t in consolidated_ticks if t['len'] >= threshold]
        if major_ticks:
            logger.info(
                f"{orientation}: Found {len(major_ticks)} major ticks "
                f"via threshold (>={threshold:.2f})."
            )
        else:
            logger.warning(f"{orientation}: Length threshold failed.")
            # Return empty list if threshold fails to find any 'major' ticks
            return []
    else:
        logger.warning(f"{orientation}: No valid lengths for differentiation.")
        return [] # Return empty list

    major_ticks.sort(key=lambda t: t['pos'])
    return major_ticks


# --- Color Extraction Helpers ---
def hex_to_hsv_tuple(hex_color):
    """Converts hex color string to an HSV tuple (OpenCV format: H 0-179, S/V 0-255)."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        logger.error(f"Invalid hex color format: '{hex_color}'")
        return None
    try:
        rgb_255 = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        # Convert RGB (0-255) to BGR (0-255) for OpenCV
        bgr_255 = np.uint8([[rgb_255[::-1]]])
        hsv = cv2.cvtColor(bgr_255, cv2.COLOR_BGR2HSV)[0][0]
        return tuple(hsv)
    except ValueError:
        logger.error(f"Invalid characters in hex color: '{hex_color}'")
        return None
    except Exception as e:
        logger.exception(f"Error converting hex '{hex_color}' to HSV: {e}")
        return None

def create_color_mask(hsv_img, target_hsv_tuple, h_tol=5, s_min=80, v_min=80):
    """Creates a mask for a given target HSV color with tolerance."""
    if target_hsv_tuple is None:
        logger.warning("Cannot create color mask: target_hsv_tuple is None.")
        # Return an empty mask of the correct size
        return np.zeros(hsv_img.shape[:2], dtype=np.uint8)

    try:
        h, s, v = target_hsv_tuple
        lower = np.array([max(0, h - h_tol), s_min, v_min])
        upper = np.array([min(179, h + h_tol), 255, 255])

        # Handle Hue wrap-around
        if h - h_tol < 0:
            lower1 = np.array([0, s_min, v_min]); upper1 = upper
            lower2 = np.array([180+(h-h_tol), s_min, v_min]); upper2 = np.array([179,255,255])
            mask = cv2.bitwise_or(cv2.inRange(hsv_img, lower1, upper1), cv2.inRange(hsv_img, lower2, upper2))
        elif h + h_tol > 179:
            lower1 = lower; upper1 = np.array([179, 255, 255])
            lower2 = np.array([0, s_min, v_min]); upper2 = np.array([(h+h_tol)-180, 255, 255])
            mask = cv2.bitwise_or(cv2.inRange(hsv_img, lower1, upper1), cv2.inRange(hsv_img, lower2, upper2))
        else:
            mask = cv2.inRange(hsv_img, lower, upper)
        return mask
    except Exception as e:
        logger.exception(f"Error creating color mask: {e}")
        return np.zeros(hsv_img.shape[:2], dtype=np.uint8)


# --- NEW Helper: Create Coordinate Map from Ticks ---
def _get_loglog_map_from_ticks(major_ticks, known_tick_values, orientation, img_height):
    """
    Creates a linear mapping log10(value) = m * pixel + b using detected ticks.

    Args:
        major_ticks: List of detected major ticks [{'pos': px, 'len': l}].
        known_tick_values: Tuple/List of known numeric values for major ticks.
        orientation: 'x' or 'y'.
        img_height: Height of the image (needed for y-coordinate conversion).

    Returns:
        Dictionary {'m': slope, 'b': intercept} or None if mapping fails.
    """
    if not major_ticks:
        logger.warning(f"Map gen: No major ticks provided for {orientation}-axis.")
        return None

    # Sort ticks by position just in case
    major_ticks.sort(key=lambda t: t['pos'])
    pixel_coords = np.array([t['pos'] for t in major_ticks])
    log_known_vals = np.log10(known_tick_values)

    # Convert Y tick pixel coordinates (image coords, 0=top) to plot coords (0=bottom)
    if orientation == 'y':
        pixel_coords = img_height - pixel_coords
        # Ensure plot coords are sorted ascending to match known values
        sort_idx = np.argsort(pixel_coords)
        pixel_coords = pixel_coords[sort_idx]
        # We assume known_tick_values are already sorted ascendingly.

    px_coords_to_fit = None
    log_vals_to_fit = None

    if len(pixel_coords) == len(log_known_vals):
        if len(pixel_coords) >= 2:
            px_coords_to_fit = pixel_coords
            log_vals_to_fit = log_known_vals
            logger.info(
                f"Map gen {orientation}: Using {len(px_coords_to_fit)} ticks matching known values."
            )
        else:
            logger.warning(f"Map gen {orientation}: Only 1 tick found. Cannot determine scale.")
            return None
    elif len(pixel_coords) >= 2:
        logger.warning(
            f"Map gen {orientation}: Mismatch between detected ticks ({len(pixel_coords)}) "
            f"and known values ({len(log_known_vals)}). Using min/max ticks."
        )
        # Use the outermost detected ticks and corresponding known values
        px_coords_to_fit = np.array([pixel_coords[0], pixel_coords[-1]])
        log_vals_to_fit = np.array([log_known_vals[0], log_known_vals[-1]])
    else:
        logger.error(
            f"Map gen {orientation}: Not enough major ticks found ({len(pixel_coords)}) "
            f"to create mapping."
        )
        return None

    # Perform linear fit: log10(value) = m * pixel + b
    try:
        # Check for sufficient variation in pixel coordinates
        if np.ptp(px_coords_to_fit) < 1e-6: # Check peak-to-peak range
             logger.error(f"Map gen {orientation}: Pixel coordinates are identical. Cannot fit line.")
             return None

        coeffs = np.polyfit(px_coords_to_fit, log_vals_to_fit, 1)
        m, b = coeffs[0], coeffs[1]
        logger.info(f"Map gen {orientation}: Fit successful (m={m:.3e}, b={b:.3e}).")
        return {'m': m, 'b': b}
    except (np.linalg.LinAlgError, ValueError) as e:
        logger.error(f"Map gen {orientation}: Linear fit failed - {e}")
        return None


# --- NEW Helper: Transform Coordinates using Fitted Map ---
def _transform_coords_loglog(points_pixels_plot_coords, map_params):
    """
    Transforms pixel coordinates (x, plot_y) to log-scaled axis values
    using fitted linear mapping parameters (log10(value) = m * pixel + b).

    Args:
        points_pixels_plot_coords: Nx2 numpy array of pixel coords (x, plot_y).
        map_params: Dictionary {'x': {'m': slope_x, 'b': intercept_x},
                               'y': {'m': slope_y, 'b': intercept_y}}

    Returns:
        Tuple (values_x, values_y) as numpy arrays, or (None, None) if mapping failed.
    """
    map_x = map_params.get('x')
    map_y = map_params.get('y')

    if map_x is None or map_y is None:
        logger.error("Coordinate transform: Incomplete mapping parameters.")
        return None, None
    if 'm' not in map_x or 'b' not in map_x or 'm' not in map_y or 'b' not in map_y:
         logger.error("Coordinate transform: Malformed mapping parameters.")
         return None, None

    px_x = points_pixels_plot_coords[:, 0]
    px_y = points_pixels_plot_coords[:, 1] # Assumes y is plot coord (0=bottom)

    try:
        # Apply mapping: value = 10**(m * pixel + b)
        log_values_x = map_x['m'] * px_x + map_x['b']
        values_x = 10**log_values_x

        log_values_y = map_y['m'] * px_y + map_y['b']
        values_y = 10**log_values_y

        # Check for NaN/Inf which can arise from extreme pixel values if fit is poor
        if not np.all(np.isfinite(values_x)) or not np.all(np.isfinite(values_y)):
            logger.warning("Non-finite values encountered during coordinate transformation.")
            # Optionally handle them (e.g., replace with NaN or clip)
            # For now, just return them and let downstream cleaning handle it.

        return values_x, values_y

    except Exception as e:
        logger.exception(f"Error during coordinate transformation: {e}")
        return None, None


# --- Main PSD Extraction Function ---
def extract_psd_from_image(
    image_path: str,
    # Standard major tick values for the target plots
    known_major_ticks_x=(10, 100, 1000),
    known_major_ticks_y=(1e-24, 1e-23, 1e-22, 1e-21, 1e-20, 1e-19),
    # Default detector colors
    detector_colors={
        "L1": "#4ba6ff", # Livingston (L1)
        "H1": "#f00000", # Hanford (H1)
        "V1": "#9b59b6"  # Virgo (V1)
    },
    # Output FrequencySeries parameters
    delta_f: float = 0.125,
    target_f_min: float = 10.0,
    target_f_max: float = 4096.0,
) -> dict | None:
    """
    Extracts PSD data curves from a plot image, using axis and major tick
    detection (scanline method) to establish coordinates. Returns data as
    PyCBC FrequencySeries

    Args:
        image_path: Path to the input image file.
        known_major_ticks_x: Tuple of known X major tick values (e.g., (10, 100, 1000)).
        known_major_ticks_y: Tuple of known Y major tick values (e.g., (1e-24, ..., 1e-19)).
        detector_colors: Dictionary mapping detector names to hex color codes.
        delta_f: Frequency resolution (delta_f) for the output FrequencySeries.
        target_f_min: Minimum frequency for the output interpolated series.
        target_f_max: Maximum frequency for the output interpolated series.

    Returns:
        A dictionary mapping detector names to FrequencySeries objects
        (real or dummy), or None if core detection/mapping fails.
    """
    img_bgra = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    alpha = (img_bgra[:, :, 3] / 255.0)[..., np.newaxis]
    bgr = img_bgra[:, :, :3]
    white_bg = np.full_like(bgr, 255)
    processed_bgr = (bgr * alpha + white_bg * (1 - alpha)).astype(np.uint8)
    img_height, img_width = processed_bgr.shape[:2]

    # --- 1. Detect Axis Lines ---
    logger.info("Detecting axis lines...")
    axis_params = find_plot_axis_lines_simplified(processed_bgr)
    if axis_params is None: logger.error("Failed to detect axis lines."); return None

    # --- 2. Detect Major Ticks ---
    logger.info("Finding major ticks...")
    dark_threshold = 80 # Use same threshold as finder
    gray = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2GRAY)
    # Generate the raw thresholded image (NO opening) for display
    ret, thresh_ticks_viz = cv2.threshold(
        gray, dark_threshold, 255, cv2.THRESH_BINARY_INV
    )

    logger.info("Finding major ticks (using scanline method)...")
    # Call finder with specific scan_offset
    major_ticks_x = find_major_ticks_by_scanline(
        processed_bgr, axis_params, orientation='horizontal',
        dark_threshold=dark_threshold, scan_offset=1
    )
    major_ticks_y = find_major_ticks_by_scanline(
        processed_bgr, axis_params, orientation='vertical',
        dark_threshold=dark_threshold, scan_offset=1
    )
    if not major_ticks_x or not major_ticks_y:
        logger.error("Failed to detect major ticks for both axes.")
        return None

    # --- 3. Establish Pixel -> Value Mapping ---
    logger.info("Creating coordinate map from ticks...")
    map_params_x = _get_loglog_map_from_ticks(
        major_ticks_x, known_major_ticks_x, 'x', img_height
    )
    map_params_y = _get_loglog_map_from_ticks(
        major_ticks_y, known_major_ticks_y, 'y', img_height
    )
    if map_params_x is None or map_params_y is None:
        logger.error("Failed to create coordinate mapping from ticks.")
        return None
    final_map_params = {'x': map_params_x, 'y': map_params_y}

    # --- 4. Extract Data Points by Color ---
    raw_psd_data = {} # Store raw {detector: (freqs, strains)}
    logger.info("Extracting points by color...")
    hsv_img = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2HSV)
    for detector, hex_color in detector_colors.items():
        target_hsv = hex_to_hsv_tuple(hex_color)
        mask = create_color_mask(hsv_img, target_hsv)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        points_pixels_img_coords = []
        for contour in contours:
            # Basic filtering: ignore tiny contours
            if cv2.contourArea(contour) < 2: continue
            # Extract all points from the contour
            for point in contour:
                points_pixels_img_coords.append(point[0]) # point[0] is (x, y)

        if not points_pixels_img_coords:
            logger.warning(f"No color points found for detector {detector}")
            continue

        points_pixels_img_coords = np.array(points_pixels_img_coords)
        # Convert to plot coordinates (y=0 at bottom)
        points_pixels_plot_coords = np.copy(points_pixels_img_coords).astype(float)
        points_pixels_plot_coords[:, 1] = img_height - points_pixels_img_coords[:, 1]

        # --- 5. Transform Color Point Coordinates ---
        freqs, strains = _transform_coords_loglog(
            points_pixels_plot_coords, final_map_params
        )
        if freqs is None or strains is None:
            logger.error(f"Coordinate transform failed for {detector}. Skipping.")
            continue

        raw_psd_data[detector] = (freqs, strains)

    import pycbc.psd
    psds ={}
    for detector, (freqs, strains) in raw_psd_data.items():
        sortorder = freqs.argsort()
        freqs = freqs[sortorder]
        strains = strains[sortorder]
        psds[detector] = pycbc.psd.from_numpy_arrays(freqs, strains**2.0,
                                                     int(target_f_max / delta_f),
                                                     delta_f, freqs[0]) 

    return psds
    


