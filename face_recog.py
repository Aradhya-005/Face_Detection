import cv2
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from scipy.spatial.distance import cosine
from supabase_helper import get_image_from_supabase, get_all_image_urls

# Initialize MTCNN and InceptionResnetV1
mtcnn = MTCNN(image_size=160, margin=20, keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

def recognize_face(img):
    """Recognize faces in an image and return embeddings."""
    pil_img = Image.fromarray(img)
    boxes, _ = mtcnn.detect(pil_img)
    
    if boxes is not None:
        faces = mtcnn(pil_img)
        embeddings = resnet(faces)
        print("Detected faces:", boxes)
        print("Face embeddings:", embeddings)
        return embeddings
    print("No faces detected.")
    return None

def compare_embeddings(embeddings1, embeddings2):
    """Compare two sets of face embeddings using cosine similarity."""
    if embeddings1 is None or embeddings2 is None:
        print("One or both embeddings are None")
        return 0

    for emb1 in embeddings1:
        for emb2 in embeddings2:
            dist = cosine(emb1, emb2)
            if dist < 0.6:  # threshold for similarity
                return True
    return False

def check_image_in_database(input_img):
    """Check if the input image matches any image in the Supabase database."""
    input_embeddings = recognize_face(input_img)
    
    if input_embeddings is None:
        print("No faces detected in the input image.")
        return None
    
    image_urls = get_all_image_urls()
    
    for image_url in image_urls:
        stored_img = get_image_from_supabase(image_url)
        if stored_img is not None:
            # Ensure image is in the correct format for comparison
            stored_img = np.array(stored_img)
            stored_embeddings = recognize_face(stored_img)
            if stored_embeddings is not None and compare_embeddings(input_embeddings, stored_embeddings):
                print(f"Image matches with record from URL: {image_url}")
                return image_url
    
    print("No match found")
    return None

def capture_image_from_webcam():
    """Capture an image from the webcam."""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return None
    
    print("Press 's' to capture an image.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            cap.release()
            return None
        
        cv2.imshow('Webcam', frame)
        
        key = cv2.waitKey(1)
        if key == ord('s'):
            print("Image captured.")
            cap.release()
            cv2.destroyAllWindows()
            return frame
        elif key == 27:  # ESC key to exit
            print("Exiting without capturing image.")
            cap.release()
            cv2.destroyAllWindows()
            return None

# Example usage
captured_image = capture_image_from_webcam()
if captured_image is not None:
    matched_image_url = check_image_in_database(captured_image)
    if matched_image_url:
        print(f"Match found: {matched_image_url}")
    else:
        print("No match found.")
