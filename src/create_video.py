import os
import argparse 
from os.path import isfile, join

import cv2
import numpy as np

from tqdm.auto import tqdm
 
def convert_frames_to_video(pathIn:str, pathOut:str, fps:int):

    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
 
    #for sorting the file names properly
    files.sort(key = lambda x: int(x.split("_")[-1].split(".")[0]))

    filename=os.path.join(pathIn, files[0])
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)

    for i in tqdm(range(len(files))):
        filename=os.path.join(pathIn, files[i])
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        #print(filename)
        #inserting the frames into an image array
        frame_array.append(img)
 
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
 
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()
    print(f"video saved: {pathOut}")
 
def main():
    parser = argparse.ArgumentParser(description="Deep Dream tutorial")
    parser.add_argument("--frames_dir", default="dream_seq", required=True, type=str, 
        help="Directory for frames created using keep_dreaming.py")
    parser.add_argument("--output_file", default="results/video.avi", type=str, help="Path of output video.")
    parser.add_argument("--fps", default=30, type=int, help="Frames per second for video")

    args = parser.parse_args()
 
    pathIn= args.frames_dir
    pathOut = args.output_file
    fps = args.fps
    convert_frames_to_video(pathIn, pathOut, fps)
 
if __name__=="__main__":
    main()