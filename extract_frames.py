import cv2
import os
import glob

def extract_sample_frames():
    # Video folder ka rasta
    video_dir = "openpose_output/video"
    # Koi ek video uthayein
    videos = glob.glob(os.path.join(video_dir, "*.mp4"))
    
    if not videos:
        print("Koi video nahi mili!")
        return
        
    sample_video = videos[0]
    print(f"Video mili: {sample_video}")
    
    # Frames save karne ke liye folder banayein
    output_dir = "sample_frames"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    cap = cv2.VideoCapture(sample_video)
    frame_count = 0
    
    while cap.isOpened() and frame_count < 10: # Sirf 10 frames nikalenge
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1
        
    cap.release()
    print(f"{frame_count} frames successfully 'sample_frames' folder mein save ho gaye!")

if __name__ == "__main__":
    extract_sample_frames()