import requests
import re
from astropy.time import Time
import astropy.units as u # Often implicitly needed with astropy.time
from urllib.parse import urljoin
import logging

# --- Logging Setup ---
# Use getLogger for better practice within modules, avoids issues if basicConfig is called elsewhere.
logger = logging.getLogger(__name__)
# Set up handler and formatter only if no handlers are already attached to this logger
if not logger.handlers:
    # Configure logger to show info level messages
    logger.setLevel(logging.INFO)
    # Create a console handler
    ch = logging.StreamHandler()
    # Create a formatter and set it for the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    # Add the handler to the logger
    logger.addHandler(ch)
# If you want basicConfig for a simple script, call it ONCE at the top level:
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Internal Helper Function ---
def _find_gwosc_image_url(gps_time: float, image_keyword: str, image_description: str) -> str | None:
    """
    Internal helper to find a specific GWOSC detector status image URL by keyword.

    Fetches the summary page for the day corresponding to the GPS time and searches
    for a PNG image link containing the specified keyword.

    Args:
        gps_time: The GPS time (seconds).
        image_keyword: The specific keyword to find within the image filename (case-insensitive).
                       (e.g., 'SPECTRUM', 'INSPIRAL_RANGE').
        image_description: A descriptive string for the image type used in logging
                           (e.g., 'spectrum', 'inspiral range').

    Returns:
        The full URL string of the target PNG image if found, otherwise None.
    """
    try:
        # 1. Convert GPS time to date string (YYYYMMDD)
        try:
            t = Time(gps_time, format='gps')
            dt_utc = t.datetime
            date_str = dt_utc.strftime('%Y%m%d')
            logger.info(f"GPS time {gps_time} corresponds to date {date_str}")
        except ValueError as e:
            logger.error(f"Error converting GPS time {gps_time}: {e}")
            return None

        # 2. Construct the URL for the specific day's directory
        # Using the base URL provided in your code:
        base_url = "https://gwosc.org/s/summary_pages/detector_status/day/"
        day_url = f"{base_url}{date_str}/"
        logger.info(f"Checking summary page URL: {day_url}")

        # 3. Fetch the HTML content of the directory page
        try:
            response = requests.get(day_url, timeout=15)
            response.raise_for_status() # Check for HTTP errors (like 404)
            html_content = response.text
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch URL {day_url}: {e}")
            return None # Network error or page not found

        # 4. Find the link to the target image using dynamic regex
        # Escape the keyword in case it contains special regex characters (unlikely here, but safer)
        # Search for href="...KEYWORD....png" case-insensitively
        pattern = rf'href="([^"]*{re.escape(image_keyword)}[^"]*\.png)"'
        match = re.search(pattern, html_content, re.IGNORECASE)

        if match:
            # 5. Construct the full, absolute image URL
            image_filename = match.group(1)
            full_image_url = urljoin(day_url, image_filename)
            logger.info(f"Found {image_description} image URL: {full_image_url}")
            return full_image_url
        else:
            # Log using the specific keyword that wasn't found
            logger.warning(f"No link containing '{image_keyword}' and ending in '.png' found on {day_url}")
            return None

    except Exception as e:
        # Catch any other unexpected errors during the process
        logger.error(f"An unexpected error occurred for GPS time {gps_time}: {e}", exc_info=True)
        return None

# --- Public Functions ---
def get_gwosc_spectrum_url(gps_time: float) -> str | None:
    """
    Retrieves the GWOSC detector status spectrum image URL for a given GPS time.

    Args:
        gps_time: The GPS time (seconds).

    Returns:
        The full URL string of the spectrum PNG image if found, otherwise None.
    """
    return _find_gwosc_image_url(
        gps_time=gps_time,
        image_keyword="SPECTRUM",
        image_description="spectrum"
    )

def get_gwosc_inspiral_range_url(gps_time: float) -> str | None:
    """
    Retrieves the GWOSC detector status inspiral range image URL for a given GPS time.

    Args:
        gps_time: The GPS time (seconds).

    Returns:
        The full URL string of the inspiral range PNG image if found, otherwise None.
    """
    return _find_gwosc_image_url(
        gps_time=gps_time,
        image_keyword="INSPIRAL_RANGE",
        image_description="inspiral range"
    )

def get_gracedb_superevents(cache=False):
    import pickle
    from ligo.gracedb.rest import GraceDb
    if cache:
        try:
            return pickle.load(open("gracedb_dump.pkl", "rb"))
        except:
            pass

    g = GraceDb()
    a = g.superevents()
    data = [k for k in a]
    pickle.dump(data, open('gracedb_dump.pkl', 'wb'))
    return data
