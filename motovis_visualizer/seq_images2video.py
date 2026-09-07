
import cv2
import os
import sys
from tqdm import tqdm
def images_to_video(image_folder, video_name, fps=25):
    video_ext = os.path.splitext(video_name)[1]
    assert video_ext in ['.mp4'], 'the postfix of video must be .mp4'
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")]
    images.sort()
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
 
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), int(fps), (width, height))
 
    for image in tqdm(images, desc='gen video'):
        video.write(cv2.imread(os.path.join(image_folder, image)))
 
    video.release()

if __name__=='__main__':
    image_folder_path = sys.argv[1]
    output_video_path = sys.argv[2]
    frames_per_second=25
    if len(sys.argv)>3:
        frames_per_second=sys.argv[3]
    images_to_video(image_folder_path, output_video_path, frames_per_second)   
        