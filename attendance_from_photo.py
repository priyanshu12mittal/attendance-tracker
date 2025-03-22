import dlib
import numpy as np
import cv2
import os
import pandas as pd
import sqlite3
import datetime
import tkinter as tk
from tkinter import filedialog

# Dlib models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

class PhotoAttendance:
    def __init__(self):
        # Initialize lists for face recognition
        self.face_features_known_list = []
        self.face_name_known_list = []
        
    def get_face_database(self):
        if os.path.exists("data/features_all.csv"):
            path_features_known_csv = "data/features_all.csv"
            csv_rd = pd.read_csv(path_features_known_csv, header=None)
            for i in range(csv_rd.shape[0]):
                features_someone_arr = []
                self.face_name_known_list.append(csv_rd.iloc[i][0])
                for j in range(1, 129):
                    if csv_rd.iloc[i][j] == '':
                        features_someone_arr.append('0')
                    else:
                        features_someone_arr.append(csv_rd.iloc[i][j])
                self.face_features_known_list.append(features_someone_arr)
            print(f"Faces in Database: {len(self.face_features_known_list)}")
            return True
        else:
            print("'features_all.csv' not found! Please run features_extraction_to_csv.py first.")
            return False
    
    @staticmethod
    def return_euclidean_distance(feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist
    
    def mark_attendance(self, name):
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()
        
        # Create attendance table if it doesn't exist
        create_table_sql = "CREATE TABLE IF NOT EXISTS attendance (name TEXT, time TEXT, date DATE, UNIQUE(name, date))"
        cursor.execute(create_table_sql)
        
        # Check if the name already has an entry for the current date
        cursor.execute("SELECT * FROM attendance WHERE name = ? AND date = ?", (name, current_date))
        existing_entry = cursor.fetchone()

        if existing_entry:
            print(f"{name} is already marked as present for {current_date}")
        else:
            current_time = datetime.datetime.now().strftime('%H:%M:%S')
            try:
                cursor.execute("INSERT INTO attendance (name, time, date) VALUES (?, ?, ?)", 
                              (name, current_time, current_date))
                conn.commit()
                print(f"✓ {name} marked as present for {current_date} at {current_time}")
            except sqlite3.IntegrityError:
                print(f"! Could not mark {name} (database constraint)")

        conn.close()
    
    def process_image(self, image_path):
        # Read the image
        img_rd = cv2.imread(image_path)
        if img_rd is None:
            print(f"Could not open or find the image: {image_path}")
            return
            
        # Resize if too large
        height, width = img_rd.shape[:2]
        max_dimension = 1200
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            img_rd = cv2.resize(img_rd, (int(width * scale), int(height * scale)))
        
        # Create a copy for drawing results
        img_with_results = img_rd.copy()
            
        # Detect faces
        faces = detector(img_rd, 1)
        
        if len(faces) == 0:
            print("No faces detected in the image")
            return
            
        print(f"Found {len(faces)} faces in the image")
        
        # List to store attendance results
        attendance_list = []
        
        # Process each face
        for i, d in enumerate(faces):
            # Get face landmarks and descriptor
            shape = predictor(img_rd, d)
            face_descriptor = face_reco_model.compute_face_descriptor(img_rd, shape)
            
            # Compare with known faces
            e_distance_list = []
            for j in range(len(self.face_features_known_list)):
                if str(self.face_features_known_list[j][0]) != '0.0':
                    e_distance = self.return_euclidean_distance(face_descriptor, self.face_features_known_list[j])
                    e_distance_list.append(e_distance)
                else:
                    e_distance_list.append(999999999)
            
            # Find the person with minimum distance
            similar_person_num = e_distance_list.index(min(e_distance_list))
            
            # If the minimum distance is less than threshold, it's a match
            if min(e_distance_list) < 0.4:
                # Mark attendance
                person_name = self.face_name_known_list[similar_person_num]
                self.mark_attendance(person_name)
                attendance_list.append(person_name)
                
                # Draw green rectangle around the face
                cv2.rectangle(img_with_results, 
                             (d.left(), d.top()), 
                             (d.right(), d.bottom()), 
                             (0, 255, 0), 2)
                
                # Draw name below the face
                cv2.putText(img_with_results, 
                           person_name, 
                           (d.left(), d.bottom() + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.8, 
                           (0, 255, 0), 
                           2)
            else:
                # Unknown face - draw red rectangle
                cv2.rectangle(img_with_results, 
                             (d.left(), d.top()), 
                             (d.right(), d.bottom()), 
                             (0, 0, 255), 2)
                
                cv2.putText(img_with_results, 
                           "Unknown", 
                           (d.left(), d.bottom() + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.8, 
                           (0, 0, 255), 
                           2)
        
        # Show summary
        print("\nAttendance Summary:")
        print("-------------------")
        print(f"Total faces detected: {len(faces)}")
        print(f"Recognized students: {len(attendance_list)}")
        if attendance_list:
            print("Students present:")
            for name in attendance_list:
                print(f"- {name}")
        
        # Display the result image
        cv2.imshow("Attendance Results", img_with_results)
        
        # Save the result image
        result_filename = "attendance_result_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
        cv2.imwrite(result_filename, img_with_results)
        print(f"\nResult image saved as: {result_filename}")
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    attendance = PhotoAttendance()
    
    # Load the face database
    if not attendance.get_face_database():
        return
    
    # Create tkinter root window and hide it
    root = tk.Tk()
    root.withdraw()
    
    # Open file dialog to select image
    image_path = filedialog.askopenfilename(
        title="Select group photo for attendance",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    
    if image_path:
        print(f"Processing image: {image_path}")
        attendance.process_image(image_path)
    else:
        print("No image selected")

if __name__ == '__main__':
    main() 