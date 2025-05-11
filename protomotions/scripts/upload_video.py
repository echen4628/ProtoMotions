import wandb
import argparse
import os
import cv2
import numpy as np

def find_latest_frame_folder(parent_dir):
    all_dirs = os.listdir(parent_dir)
    all_dirs = [path for path in all_dirs if os.path.isdir(os.path.join(parent_dir, path))]
    if not all_dirs:
        raise RuntimeError("No subdirectories found to use as image folders.")
    return max(all_dirs)

def create_video_from_folder(project_name, img_folder, fps):
    run = wandb.init(project=project_name)

    # Taken from base_simulator/simulator.py
    from moviepy.editor import ImageSequenceClip
    
    image_dir = img_folder
    images = sorted([
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.endswith('.png')
    ])

    clip = ImageSequenceClip(images, fps=fps)
    clip.write_videofile(
        f"{img_folder}.mp4",
        codec='libx264',
        audio=False,
        threads=32,
        preset='veryfast',
        ffmpeg_params=[
            '-profile:v', 'main',
            '-level', '4.0',
            '-pix_fmt', 'yuv420p', 
            '-movflags', '+faststart',
            '-crf', '23',
            '-x264-params', 'keyint=60:min-keyint=30'
        ]
    )
    print(f"Video saved to {img_folder}.mp4")
    wandb.log({"rollout": wandb.Video(f"{img_folder}.mp4", format='gif')})
    run.finish()
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Upload a video to W&B from image frames.")
    parser.add_argument("--project", type=str, required=True, help="W&B project name")
    parser.add_argument("--folder", type=str, default=None, help="Path to folder of image frames")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the video")

    args = parser.parse_args()
    base_output = "output/renderings"
    folder_path = os.path.join(base_output, args.folder if args.folder else find_latest_frame_folder(base_output))
    create_video_from_folder(args.project, folder_path, args.fps)