import requests
import numpy as np
import cv2
from PIL import Image
from supabase import create_client

# Supabase configuration
url = "actual database url"  # Replace with your actual Supabase URL
key = "actual Supabase key"  # Replace with your actual Supabase key
supabase = create_client(url, key)

def get_image_from_url(url):
    """Download and decode an image from a URL."""
    try:
        resp = requests.get(url)
        resp.raise_for_status()  # Raise an exception for HTTP errors
        img_array = np.asarray(bytearray(resp.content), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # Use IMREAD_COLOR for standard color images
        
        if img is None:
            print(f"Error: Failed to decode image from URL: {url}")
            return None
        return img
    except requests.RequestException as e:
        print(f"Error downloading image from URL: {url}. Exception: {e}")
        return None

def convert_bgr_to_rgb(img):
    """Convert BGR image to RGB."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def get_image_from_supabase(image_url):
    """Download an image from Supabase Storage and convert it to RGB."""
    img = get_image_from_url(image_url)
    if img is not None:
        img_rgb = convert_bgr_to_rgb(img)
        return img_rgb
    else:
        print(f"Error: Unable to get image from Supabase URL: {image_url}")
        return None

def get_all_image_urls():
    """Fetch all image URLs from Supabase."""
    try:
        response = supabase.table("User").select("photo").execute()
        
        if response.error:
            print(f"Error fetching data from Supabase: {response.error}")
            return []
        
        if not response.data:
            print("No data found in the response.")
            return []
        
        image_urls = [record['photo'] for record in response.data if 'photo' in record]
        
        if not image_urls:
            print("No image URLs found in the database.")
        
        return image_urls

    except Exception as e:
        print(f"Error fetching image URLs: {e}")
        return []

# Test the functions
def test_functions():
    # Replace with an actual image URL from your Supabase Storage
    test_img_url = 'https://hgbhohchavurggrbwwta.supabase.co/storage/v1/object/public/images/sample.jpg'
    img = get_image_from_supabase(test_img_url)
    if img is not None:
        print("Image fetched and converted successfully.")
        # Optionally save or display the image
        pil_img = Image.fromarray(img)
        pil_img.save('test_image.jpg')  # Save the image for verification
        pil_img.show()  # Display the image
    else:
        print("Failed to fetch or convert image.")

if __name__ == "__main__":
    test_functions()
