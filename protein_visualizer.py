import cv2
import numpy as np
import mediapipe as mp
import subprocess
import requests
import os

# Function to fetch PDB file from RCSB PDB
def fetch_pdb(pdb_id):
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    response = requests.get(url)
    if response.status_code == 200:
        pdb_file = f"{pdb_id}.pdb"
        with open(pdb_file, "wb") as f:
            f.write(response.content)
        print(f"PDB file {pdb_file} downloaded successfully.")
        return pdb_file
    else:
        print(f"Failed to fetch PDB file for {pdb_id}.")
        return None

# Function to visualize protein structure using PyMOL (cartoon representation)
def visualize_protein(pdb_file):
    # Use PyMOL to render the protein structure
    pymol_script = f"""
    load {pdb_file}
    show cartoon
    color red, ss h  # Alpha helices
    color yellow, ss s  # Beta sheets
    color green, ss l+''  # Loops
    set ray_opaque_background, off  # Transparent background
    png protein_structure.png, width=800, height=600, dpi=300
    quit
    """
    with open("pymol_script.pml", "w") as f:
        f.write(pymol_script)
    
    # Run PyMOL in command-line mode
    subprocess.run(["pymol", "-c", "pymol_script.pml"])
    print("Protein structure rendered and saved as 'protein_structure.png'.")

# Function to overlay protein structure on camera feed with hand interaction
def overlay_protein_on_camera(protein_image):
    # Check if the protein image exists
    if not os.path.exists(protein_image):
        print(f"Error: File '{protein_image}' not found.")
        return
    
    # Load the protein image with transparency
    protein_img = cv2.imread(protein_image, cv2.IMREAD_UNCHANGED)
    if protein_img is None:
        print(f"Error: Failed to load image '{protein_image}'.")
        return
    
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
    
    # Open camera feed
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Initialize transformation variables
    angle = 0  # Rotation angle
    scale = 1.0  # Scaling factor
    dx, dy = 0, 0  # Translation offsets
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)
        
        # Detect hands
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Process hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get the coordinates of the index finger tip (landmark 8)
                index_finger_tip = hand_landmarks.landmark[8]
                h, w, _ = frame.shape
                x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                
                # Draw a circle at the index finger tip
                cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
                
                # Map hand movement to protein transformation
                dx = x - w // 2  # Horizontal translation
                dy = y - h // 2  # Vertical translation
                angle = (x / w) * 360  # Rotation angle based on horizontal position
                scale = 1.0 + (y / h)  # Scaling factor based on vertical position
        
        # Apply transformations to the protein image
        rows, cols, _ = protein_img.shape
        M = cv2.getRotationMatrix2D((cols // 2, rows // 2), angle, scale)
        rotated_protein = cv2.warpAffine(protein_img, M, (cols, rows))
        
        # Translate the protein image
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        translated_protein = cv2.warpAffine(rotated_protein, M, (cols, rows))
        
        # Overlay protein image on camera feed
        overlay = cv2.resize(translated_protein, (frame.shape[1], frame.shape[0]))
        alpha = overlay[:, :, 3] / 255.0  # Alpha channel
        for c in range(0, 3):
            frame[:, :, c] = frame[:, :, c] * (1 - alpha) + overlay[:, :, c] * alpha
        
        # Display the frame
        cv2.imshow("Protein AR Overlay", frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Main function
def main():
    pdb_id = input("Enter PDB ID (e.g., 1A2C): ").strip().upper()
    pdb_file = fetch_pdb(pdb_id)
    
    if not pdb_file:
        return
    
    # Visualize protein structure using PyMOL
    visualize_protein(pdb_file)
    
    # Overlay protein structure on camera feed with hand interaction
    overlay_protein_on_camera("protein_structure.png")

if __name__ == "__main__":
    main()