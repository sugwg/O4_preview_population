import cv2
import numpy as np
import logging
import os
from scipy.interpolate import interp1d

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

# --- Reusable Image Processing Functions ---

def find_plot_axis_lines_simplified(img):
    """
    Detects the main horizontal (bottom) and vertical (left) axis lines
    using Hough transform on a mask of grey pixels.
    """
    if img is None or len(img.shape) < 2:
        logger.error("Invalid input image provided.")
        return None
    img_height, img_width = img.shape[:2]

    try:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lower_grey, upper_grey = 100, 210
        grey_mask = cv2.inRange(gray_img, lower_grey, upper_grey)
        kernel = np.ones((3, 3), np.uint8)
        closed_grey_mask = cv2.morphologyEx(grey_mask, cv2.MORPH_CLOSE, kernel)
    except Exception as e:
        logger.exception(f"Preprocessing error during axis detection: {e}")
        return None

    min_axis_length = min(img_width, img_height) * 0.7
    lines = cv2.HoughLinesP(
        closed_grey_mask, rho=1, theta=np.pi / 180, threshold=30,
        minLineLength=min_axis_length, maxLineGap=20
    )

    if lines is None:
        logger.warning("HoughLinesP did not detect long lines for axes.")
        return None

    horizontal_lines, vertical_lines = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1)
        if abs(angle) < np.deg2rad(3):
            horizontal_lines.append({'y': (y1+y2)/2, 'x1': min(x1,x2), 'x2': max(x1,x2), 'len': abs(x2-x1)})
        elif abs(abs(angle) - np.pi/2) < np.deg2rad(3):
            vertical_lines.append({'x': (x1+x2)/2, 'y1': min(y1,y2), 'y2': max(y1,y2), 'len': abs(y2-y1)})

    candidate_xaxis = [l for l in horizontal_lines if l['y'] > img_height * 0.5]
    if not candidate_xaxis: return None
    best_xaxis = max(candidate_xaxis, key=lambda l: l['len'])
    
    candidate_yaxis = [l for l in vertical_lines if l['x'] < img_width * 0.25]
    if not candidate_yaxis: return None
    best_yaxis = max(candidate_yaxis, key=lambda l: l['len'])

    adjusted_x_start = best_yaxis['x']
    adjusted_y_end = best_xaxis['y']

    return {
        'x_axis': (adjusted_x_start, best_xaxis['x2'], adjusted_y_end),
        'y_axis': (best_yaxis['y1'], adjusted_y_end, adjusted_x_start)
    }

def find_major_ticks_by_scanline(
    img_bgr, axis_params, orientation='horizontal', scan_offset=1,
    max_scan_depth=8, min_tick_length=2, consolidation_dist=5,
    major_tick_len_threshold=1.4, dark_threshold=120,
    assume_all_ticks_are_major=False
):
    """
    Finds tick marks by scanning pixel lines. The scan range is slightly extended
    to catch ticks at the very edge of the plot.
    """
    if not all([img_bgr is not None, axis_params]): return []
    img_height, img_width = img_bgr.shape[:2]
    raw_ticks, consolidated_ticks = [], []

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh_ticks = cv2.threshold(gray, dark_threshold, 255, cv2.THRESH_BINARY_INV)

    x_start, x_end, y_axis_pos = map(float, axis_params['x_axis'])
    y_start, y_end, x_axis_pos = map(float, axis_params['y_axis'])

    scan_buffer = 10 # Increased buffer for more robust tick detection

    if orientation == 'horizontal':
        y_scan_start = int(round(y_axis_pos)) + scan_offset
        for x in range(int(round(x_start)), int(round(x_end)) + scan_buffer):
            if not (0 <= x < img_width): continue
            run_length = 0
            for y in range(y_scan_start, min(y_scan_start + max_scan_depth, img_height)):
                if thresh_ticks[y, x] == 255: run_length += 1
                else: break
            if run_length >= min_tick_length: raw_ticks.append({'pos': float(x), 'len': run_length})
    else: # vertical
        x_scan_start = int(round(x_axis_pos)) - scan_offset
        for y in range(int(round(y_start)) - scan_buffer, int(round(y_end)) + scan_buffer):
            if not (0 <= y < img_height): continue
            run_length = 0
            for x in range(x_scan_start, max(0, x_scan_start - max_scan_depth), -1):
                if not (0 <= x < img_width): break
                if thresh_ticks[y, x] == 255: run_length += 1
                else: break
            if run_length >= min_tick_length: raw_ticks.append({'pos': float(y), 'len': run_length})

    if not raw_ticks: 
        logger.warning(f"Scanline found no raw ticks for {orientation} axis.")
        return []
        
    raw_ticks.sort(key=lambda t: t['pos'])
    
    current_group = [raw_ticks[0]]
    for i in range(1, len(raw_ticks)):
        if raw_ticks[i]['pos'] - current_group[-1]['pos'] <= consolidation_dist:
            current_group.append(raw_ticks[i])
        else:
            consolidated_ticks.append({'pos': np.mean([t['pos'] for t in current_group]), 'len': max(t['len'] for t in current_group)})
            current_group = [raw_ticks[i]]
    consolidated_ticks.append({'pos': np.mean([t['pos'] for t in current_group]), 'len': max(t['len'] for t in current_group)})

    if not consolidated_ticks:
        logger.warning(f"No ticks found for {orientation} axis after consolidation.")
        return []

    if assume_all_ticks_are_major:
        major_ticks = consolidated_ticks
        logger.info(f"{orientation}: Assuming all {len(consolidated_ticks)} consolidated ticks are major ticks.")
    else:
        lengths = np.array([t['len'] for t in consolidated_ticks])
        threshold = max(np.median(lengths) * major_tick_len_threshold, min_tick_length * 1.1)
        major_ticks = [t for t in consolidated_ticks if t['len'] >= threshold]
        logger.info(
            f"{orientation}: Found {len(major_ticks)} major ticks from {len(consolidated_ticks)} "
            f"candidates via length threshold (>={threshold:.2f})."
        )

    if not major_ticks:
        logger.warning(f"No major ticks identified for {orientation} axis.")
        return []

    major_ticks.sort(key=lambda t: t['pos'])
    return major_ticks

def hex_to_hsv_tuple(hex_color):
    """Converts a hex color string to an HSV tuple for OpenCV."""
    hex_color = hex_color.lstrip('#')
    rgb_255 = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    bgr_255 = np.uint8([[rgb_255[::-1]]])
    return tuple(cv2.cvtColor(bgr_255, cv2.COLOR_BGR2HSV)[0][0])

def create_color_mask(hsv_img, target_hsv_tuple, h_tol=15, s_min=80, v_min=80):
    """
    Creates a mask for a given target HSV color with more lenient tolerances.
    """
    h, s, v = target_hsv_tuple
    lower = np.array([max(0, h - h_tol), s_min, v_min])
    upper = np.array([min(179, h + h_tol), 255, 255])
    # Handle Hue wrap-around for red/orange colors near 0
    if h - h_tol < 0:
        lower1 = np.array([0, s_min, v_min]); upper1 = upper
        lower2 = np.array([180 + (h - h_tol), s_min, v_min]); upper2 = np.array([179, 255, 255])
        mask = cv2.bitwise_or(cv2.inRange(hsv_img, lower1, upper1), cv2.inRange(hsv_img, lower2, upper2))
    else:
        mask = cv2.inRange(hsv_img, lower, upper)
    return mask

def _get_linear_map_from_ticks(major_ticks, known_tick_values, orientation, img_height):
    """Creates a linear mapping from pixel coordinates to data values: value = m * pixel + b."""
    if not major_ticks:
        logger.warning(f"Map gen: No major ticks for {orientation}-axis.")
        return None
    major_ticks.sort(key=lambda t: t['pos'])
    pixel_coords = np.array([t['pos'] for t in major_ticks])
    
    if orientation == 'y':
        pixel_coords = img_height - pixel_coords
        pixel_coords = np.sort(pixel_coords)

    px_coords_to_fit, vals_to_fit = None, None
    if len(pixel_coords) == len(known_tick_values):
        if len(pixel_coords) >= 2:
            px_coords_to_fit = pixel_coords
            vals_to_fit = np.array(known_tick_values)
        else:
            logger.warning(f"Map gen {orientation}: Only 1 tick found. Cannot fit."); return None
    elif len(pixel_coords) >= 2:
        logger.warning(
            f"Map gen {orientation}: Mismatch in ticks ({len(pixel_coords)}) and known values "
            f"({len(known_tick_values)}). Using min/max ticks for mapping."
        )
        px_coords_to_fit = np.array([pixel_coords[0], pixel_coords[-1]])
        vals_to_fit = np.array([known_tick_values[0], known_tick_values[-1]])
    else:
        logger.error(f"Map gen {orientation}: Not enough ticks ({len(pixel_coords)}) to create a map.")
        return None

    try:
        coeffs = np.polyfit(px_coords_to_fit, vals_to_fit, 1)
        m, b = coeffs[0], coeffs[1]
        logger.info(f"Map gen {orientation}: Linear fit successful (m={m:.3e}, b={b:.3e}).")
        return {'m': m, 'b': b}
    except Exception as e:
        logger.error(f"Map gen {orientation}: Linear fit failed - {e}")
        return None

def _transform_coords_linear(points_pixels_plot_coords, map_params):
    """Transforms pixel coordinates to data values using a linear map."""
    map_x, map_y = map_params.get('x'), map_params.get('y')
    if not all([map_x, map_y]): return None, None

    px_x = points_pixels_plot_coords[:, 0]
    px_y = points_pixels_plot_coords[:, 1]

    values_x = map_x['m'] * px_x + map_x['b']
    values_y = map_y['m'] * px_y + map_y['b']
    return values_x, values_y
        
# --- Main Data Extraction Function ---

def extract_range_data_from_file(
    image_path: str,
    detector_colors={
        "H1": "#f07d02", # Red-ish Orange for Hanford
        "L1": "#4ba6ff", # Blue for Livingston
        "V1": "#9b59b6"  # Purple for Virgo
    },
) -> dict | None:
    """
    Extracts time series data from a BNS Inspiral Range plot image file.

    Args:
        image_path: The file path to the plot image.
        detector_colors: A dictionary mapping detector names to their hex color codes.

    Returns:
        A dictionary mapping each detector name to a tuple containing two 
        numpy arrays with a 1-to-1 mapping: (times, ranges). Returns None on failure.
    """
    if not os.path.exists(image_path):
        logger.error(f"Image file not found at: {image_path}")
        return None
        
    try:
        img_bgra = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img_bgra is None:
            logger.error(f"OpenCV could not read image file: {image_path}"); return None

        if img_bgra.shape[2] == 4:
            alpha = (img_bgra[:, :, 3] / 255.0)[..., np.newaxis]
            bgr = img_bgra[:, :, :3]
            white_bg = np.full_like(bgr, 255)
            processed_bgr = (bgr * alpha + white_bg * (1 - alpha)).astype(np.uint8)
        else:
            processed_bgr = img_bgra
            
        img_height, img_width = processed_bgr.shape[:2]
    except Exception as e:
        logger.exception(f"Failed during image loading and preprocessing: {e}"); return None

    axis_params = find_plot_axis_lines_simplified(processed_bgr)
    if axis_params is None: 
        logger.error("Failed to detect axis lines."); return None

    major_ticks_x = find_major_ticks_by_scanline(
        processed_bgr, axis_params, orientation='horizontal', assume_all_ticks_are_major=False
    )
    major_ticks_y = find_major_ticks_by_scanline(
        processed_bgr, axis_params, orientation='vertical', assume_all_ticks_are_major=True
    )
    
    if not major_ticks_x or not major_ticks_y:
        logger.error("Failed to detect major ticks for one or both axes.")
        return None

    known_major_ticks_x = np.arange(0, 25, 2)
    
    num_y_ticks = len(major_ticks_y)
    
    # --- NEW: Determine Y-axis tick spacing based on the number of ticks found ---
    if num_y_ticks <= 7:
        y_tick_spacing = 25
        logger.info(f"Found {num_y_ticks} Y-ticks. Assuming spacing of {y_tick_spacing}.")
    else:
        y_tick_spacing = 20
        logger.info(f"Found {num_y_ticks} Y-ticks. Assuming default spacing of {y_tick_spacing}.")

    known_major_ticks_y = np.arange(start=0, stop=num_y_ticks * y_tick_spacing, step=y_tick_spacing)
    if num_y_ticks > 0:
        logger.info(f"Generated known Y-ticks with max value {known_major_ticks_y[-1] if num_y_ticks > 0 else 0}")

    map_params_x = _get_linear_map_from_ticks(major_ticks_x, known_major_ticks_x, 'x', img_height)
    map_params_y = _get_linear_map_from_ticks(major_ticks_y, known_major_ticks_y, 'y', img_height)
    if not all([map_params_x, map_params_y]):
        logger.error("Failed to create coordinate mapping from ticks."); return None
    final_map_params = {'x': map_params_x, 'y': map_params_y}

    extracted_data = {}
    hsv_img = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2HSV)
    for detector, hex_color in detector_colors.items():
        target_hsv = hex_to_hsv_tuple(hex_color)
        mask = create_color_mask(hsv_img, target_hsv)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            logger.warning(f"No color contours found for detector {detector}"); continue
            
        all_points = np.vstack([c for c in contours if cv2.contourArea(c) > 1])
        points_pixels_img_coords = all_points[:, 0, :]

        # Average in Pixel Space to prevent float issues
        sorted_points = points_pixels_img_coords[points_pixels_img_coords[:, 0].argsort()]
        unique_x_coords = np.unique(sorted_points[:, 0])
        median_y_coords = [
            np.median(sorted_points[sorted_points[:, 0] == x, 1]) 
            for x in unique_x_coords
        ]
        clean_pixel_coords_img = np.array(
            list(zip(unique_x_coords, median_y_coords)), dtype=float
        )

        # Convert clean image pixel coordinates to plot coordinates (y=0 at bottom)
        clean_pixel_coords_plot = np.copy(clean_pixel_coords_img)
        clean_pixel_coords_plot[:, 1] = img_height - clean_pixel_coords_img[:, 1]
        
        # Transform the final, clean pixel coordinates to (time, range) values
        times, ranges = _transform_coords_linear(clean_pixel_coords_plot, final_map_params)
        if times is None or ranges is None:
            logger.error(f"Coordinate transform failed for {detector}. Skipping."); continue
        
        extracted_data[detector] = (times, ranges)
        logger.info(f"Successfully extracted and processed {len(times)} points for {detector}.")

    return extracted_data
    
def get_ranges(time):

    from gworacle import query
    from astropy.utils.data import download_file
    import numpy
    import astropy.time

    
    fname = download_file(query.get_gwosc_inspiral_range_url(time), cache=True)
    d = extract_range_data_from_file(fname)

    t = astropy.time.Time(time, scale='utc', format='gps')
    thour = t.datetime.hour + t.datetime.minute / 60 + t.datetime.second / 3600
    
    return {k: numpy.interp(thour, d[k][0], d[k][1]) for k in d}
