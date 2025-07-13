import requests
import re
import os
import logging
from urllib.parse import urljoin, urlparse
import shutil # Needed for copying file from cache (though not used in this version)
import json   # Needed to load JSON from file

# Added Astropy import with try-except
try:
    from astropy.utils.data import download_file
    # URLError includes HTTPError in newer Python/Astropy versions
    from urllib.error import URLError
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False
    # Logging setup happens below
    print("WARNING: astropy not found. Install with 'pip install astropy' to enable file download caching.")

# --- Logging Setup --- (Identical to user's provided code)
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO); ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter); logger.addHandler(ch)


def get_gracedb_skymap(
    event_id: str,
    map_type: str = "bayestar",
    use_cache: bool = True # Option to control caching for BOTH downloads
) -> str | None:
    """
    Downloads a skymap FITS file from GraceDB for a given superevent using the API,
    leveraging astropy's download cache for both the file list and the FITS file.

    Args:
        event_id: The GraceDB event ID (case-sensitive).
        map_type: 'bayestar' or 'bilby' (case-insensitive).
        use_cache: If True (and astropy is installed), use astropy's download
                   cache for file list and FITS file. Defaults to True.

    Returns:
        The local path to the downloaded/cached FITS file if successful,
        otherwise None. Note: This path will be inside the astropy cache.
    """
    if not ASTROPY_AVAILABLE:
         logger.error("Astropy is required for downloading files with caching.")
         # Optionally could add a non-cached fallback using requests here
         return None

    # --- API Query for File List using download_file ---
    if not event_id: logger.error("Event ID cannot be empty."); return None
    api_files_url = f"https://gracedb.ligo.org/api/superevents/{event_id}/files/"
    logger.info(f"Querying/Caching GraceDB API file list: {api_files_url}")

    can_cache = ASTROPY_AVAILABLE and use_cache
    file_list_cache_path = None
    file_data = None

    try:
        # Download the JSON file list, using cache if enabled/available
        file_list_cache_path = download_file(
            api_files_url,
            cache=can_cache,
            show_progress=False,
            pkgname='gworacle_cache', # Consistent cache sub-directory
            timeout=30
        )
        logger.debug(f"File list cached/retrieved at: {file_list_cache_path}")

        # Read the JSON data from the downloaded/cached file
        with open(file_list_cache_path, 'r') as f:
            file_data = json.load(f)

    # Catch specific download errors from astropy/urllib
    except URLError as e:
        logger.error(f"Failed to download file list from {api_files_url}: {e}")
        return None
    except TimeoutError as e:
         logger.error(f"Timeout downloading file list from {api_files_url}: {e}")
         return None
    # Catch JSON parsing errors
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON file list from {file_list_cache_path}: {e}")
        # Optionally log file content snippet if path exists
        if file_list_cache_path and os.path.exists(file_list_cache_path):
             try:
                  with open(file_list_cache_path, 'r') as f_err:
                       logger.debug(f"File list snippet: {f_err.read(500)}")
             except Exception: pass # Ignore errors reading snippet
        return None
    # Catch other potential errors (e.g., file IO)
    except Exception as e:
        logger.exception(f"An unexpected error occurred getting file list: {e}")
        return None

    # --- Filename Selection (Identical logic as user's provided code) ---
    target_filename = None; filenames = []
    if isinstance(file_data, dict): filenames = list(file_data.keys())
    elif isinstance(file_data, list):
        if file_data and isinstance(file_data[0], dict) and 'filename' in file_data[0]: filenames = [item['filename'] for item in file_data if 'filename' in item]
        elif file_data and isinstance(file_data[0], str): filenames = file_data
        else: logger.error("Unrecognized JSON list structure."); return None
    else: logger.error("Unexpected JSON structure."); return None
    if not filenames: logger.warning(f"No files listed for {event_id}."); return None

    map_type_lower = map_type.lower(); logger.info(f"Searching {len(filenames)} files for '{map_type_lower}' FITS...")
    found_gz = None; found_fits = None
    for fname in filenames:
        fname_lower = fname.lower()
        if map_type_lower in fname_lower:
            if fname_lower.endswith('multiorder.fits.gz') or re.search(r'\.multiorder\.fits\.gz,\d+$', fname_lower):
                if found_gz is None: found_gz = fname
            elif (fname_lower.endswith('multiorder.fits') or re.search(r'\.multiorder\.fits,\d+$', fname_lower)):
                if found_fits is None: found_fits = fname
            elif (fname_lower.endswith('.fits') or re.search(r'\.fits,\d+$', fname_lower)):
                if found_fits is None: found_fits = fname
        if found_gz: break # Prioritize .gz
    if found_gz: target_filename = found_gz; logger.info(f"Found prioritized '.fits.gz': {target_filename}")
    elif found_fits: target_filename = found_fits; logger.info(f"Found fallback '.fits': {target_filename}")
    else: logger.warning(f"No '{map_type}' FITS skymap found for {event_id}."); return None

    # --- Download Selected FITS File using download_file ---
    download_url = f"{api_files_url.rstrip('/')}/{target_filename}"; logger.info(f"Download URL: {download_url}")

    logger.info(
        f"Attempting FITS download using astropy utility (cache={'enabled' if can_cache else 'disabled'})..."
    )
    try:
        # Download the FITS file, using cache if enabled/available
        fits_cache_path = download_file(
            download_url,
            cache=can_cache,
            show_progress=False,
            pkgname='gworacle_cache', # Keep cache organized
            timeout=60 # Potentially longer timeout for larger FITS files
        )
        logger.info(f"FITS file downloaded/cached by astropy at: {fits_cache_path}")
        # --- Return the path to the file IN THE CACHE ---
        # Modification from user code: No longer copying, just return cache path
        return fits_cache_path

    except URLError as e:
        logger.error(f"Failed to download FITS file from {download_url}: {e}")
        return None
    except TimeoutError as e:
         logger.error(f"Timeout downloading FITS file from {download_url}: {e}")
         return None
    except Exception as e:
        logger.exception(f"An unexpected error occurred during FITS download: {e}")
        return None
