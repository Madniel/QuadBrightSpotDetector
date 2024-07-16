from functools import wraps
import logging


def error_handler(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            logging.basicConfig(filename='brightness_patch_detector_error.log',
                                level=logging.ERROR,
                                format='%(asctime)s %(levelname)s: %(message)s')
            logging.error(f"An error occurred in function '{func.__name__}': {e}")
            print(f"An error occurred in function '{func.__name__}': {e}")

    return wrapper
