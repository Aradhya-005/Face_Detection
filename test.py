import requests
import numpy as np
import cv2
from PIL import Image

def get_image_from_url(url):
    """Download and decode an image from a URL."""
    try:
        resp = requests.get(url)
        resp.raise_for_status()  # Raise an exception for HTTP errors
        img_array = np.asarray(bytearray(resp.content), dtype=np.uint8)
        img = cv2.imdecode(img_array, -1)
        
        if img is None:
            print(f"Error: Failed to decode image from URL: {url}")
        return img
    except Exception as e:
        print(f"Error downloading image from URL: {url}. Exception: {e}")
        return None

def convert_bgr_to_rgb(img):
    """Convert BGR image to RGB."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Test URL
test_img_url ="path _of_img"
# Get image
test_img = get_image_from_url(test_img_url)

if test_img is not None:
    # Convert BGR to RGB
    rgb_img = convert_bgr_to_rgb(test_img)
    
    # Convert to PIL Image and display
    pil_img = Image.fromarray(rgb_img)
    pil_img.show()
else:
    print("Error: Unable to process the image.")
