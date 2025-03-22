import dlib
import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import simpledialog, filedialog

# Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()

class GroupPhotoRegistration:
    def __init__(self):
        self.path_photos_from_camera = "data/data_faces_from_camera/"
        self.current_face_count = 0
        
        # Create directory if it doesn't exist
        if not os.path.exists(self.path_photos_from_camera):
            os.makedirs(self.path_photos_from_camera)
            
        # Get existing face count
        if os.listdir(self.path_photos_from_camera):
            person_list = os.listdir(self.path_photos_from_camera)
            person_num_list = []
            for person in person_list:
                person_order = person.split('_')[1].split('_')[0]
                person_num_list.append(int(person_order))
            self.current_face_count = max(person_num_list)
        
    def extract_faces(self, image_path):
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not open or find the image: {image_path}")
            return
        
        # Resize if too large
        height, width = img.shape[:2]
        max_dimension = 1200
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            img = cv2.resize(img, (int(width * scale), int(height * scale)))
        
        # Create a copy for drawing
        img_display = img.copy()
        
        # Detect faces
        faces = detector(img, 1)
        
        if len(faces) == 0:
            print("No faces detected in the image")
            return
        
        print(f"Found {len(faces)} faces in the image")
        
        # Initialize tkinter
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        # Process each face
        for i, face in enumerate(faces):
            # Draw rectangle on display image
            cv2.rectangle(img_display, (face.left(), face.top()), 
                         (face.right(), face.bottom()), (0, 255, 0), 2)
            
            # Put face number
            cv2.putText(img_display, f"Face #{i+1}", (face.left(), face.top() - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        # Show image with numbered faces
        cv2.imshow("Detected Faces", img_display)
        cv2.waitKey(1000)  # Show for a second before asking for names
        
        # Ask for name for each face and save
        for i, face in enumerate(faces):
            # Highlight current face
            img_current = img_display.copy()
            cv2.rectangle(img_current, (face.left(), face.top()), 
                         (face.right(), face.bottom()), (0, 0, 255), 3)
            cv2.putText(img_current, f"Face #{i+1} - Naming this person", 
                       (face.left(), face.top() - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow("Detected Faces", img_current)
            cv2.waitKey(100)
            
            # Ask for name
            name = simpledialog.askstring("Input", f"Enter name for Face #{i+1}:", parent=root)
            
            if name:
                # Create directory for this person
                self.current_face_count += 1
                person_dir = f"{self.path_photos_from_camera}person_{self.current_face_count}_{name}"
                os.makedirs(person_dir, exist_ok=True)
                
                # Extract face with some margin
                height = face.height()
                width = face.width()
                margin_h = int(height * 0.5)
                margin_w = int(width * 0.5)
                
                top = max(0, face.top() - margin_h)
                bottom = min(img.shape[0], face.bottom() + margin_h)
                left = max(0, face.left() - margin_w)
                right = min(img.shape[1], face.right() + margin_w)
                
                face_img = img[top:bottom, left:right]
                
                # Save multiple versions of the face (with slight variations for better recognition)
                for j in range(5):
                    filename = f"{person_dir}/img_face_{j+1}.jpg"
                    
                    # Save original for first image
                    if j == 0:
                        cv2.imwrite(filename, face_img)
                        continue
                        
                    # Create variations: brightness, contrast, slight rotation
                    if j == 1:
                        # Brighter
                        adjusted = cv2.convertScaleAbs(face_img, alpha=1.1, beta=10)
                    elif j == 2:
                        # Darker
                        adjusted = cv2.convertScaleAbs(face_img, alpha=0.9, beta=-10)
                    elif j == 3:
                        # Slight rotation left
                        center = (face_img.shape[1] // 2, face_img.shape[0] // 2)
                        M = cv2.getRotationMatrix2D(center, 5, 1.0)
                        adjusted = cv2.warpAffine(face_img, M, (face_img.shape[1], face_img.shape[0]))
                    else:
                        # Slight rotation right
                        center = (face_img.shape[1] // 2, face_img.shape[0] // 2)
                        M = cv2.getRotationMatrix2D(center, -5, 1.0)
                        adjusted = cv2.warpAffine(face_img, M, (face_img.shape[1], face_img.shape[0]))
                    
                    cv2.imwrite(filename, adjusted)
                
                print(f"Saved face of {name}")
            else:
                print(f"Skipped Face #{i+1}")
        
        cv2.destroyAllWindows()
        root.destroy()
        
        print("Face registration complete! Now run features_extraction_to_csv.py")

if __name__ == "__main__":
    registration = GroupPhotoRegistration()
    
    # Ask for image file
    root = tk.Tk()
    root.withdraw()
    image_path = filedialog.askopenfilename(title="Select group photo", 
                                          filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    
    if image_path:
        registration.extract_faces(image_path)
    else:
        print("No image selected") 