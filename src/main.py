# coding: utf-8
import os

from ultralytics import YOLO
from argparse import ArgumentParser
import torch
import cv2
import re
from paddleocr import PaddleOCR
import math
import time
from torch.cuda.amp import autocast

import cProfile
import pstats

profiler = cProfile.Profile()

def distance(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def parse_arguments():
    parser = ArgumentParser(description="Inference script for detecting and recognising license plates using pre-trained YOLO and OCR model.")

    parser.add_argument("-i", "--input", type=str, required=False, help="Path to the input file.")
    parser.add_argument("-o", "--output", type=str, required=False, help="Path to the output file.")
    parser.add_argument("-y", "--yolo", type=str, required=True, help="Path to the YOLO model.")
    parser.add_argument("-p", "--profiler", type=bool, required=False, help="If set to true, will profile the run.")

    return parser.parse_args()

def run_detection_video(
    yolo: YOLO,
    ocr,
    video_path: str,
    output_path: str,
    profile: bool
) -> None:
    print("Creating video capture object ...")
    if profile:
        profiler = cProfile.Profile()
        profiler.enable()

    if video_path == None:  # input from default system camera
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            exit("ERROR: Unable to read input data from camera source!")
            return
    else:                   # input from file
        video_capture = cv2.VideoCapture(video_path)    
        if not video_capture.isOpened():
            print(f"ERROR: Unable to open video file for reading {video_path}.")
            return
        
    frame_rate = int(video_capture.get(cv2.CAP_PROP_FPS))
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frame_interval = 1.0 / frame_rate
    start = time.time()

    
    print("Video capture object created.")
    if output_path is not None:
        print("Creating video writer object ...")
        output_file_path = os.path.join(output_path, "output.avi")
        video_writer = cv2.VideoWriter(
            output_file_path,
            cv2.VideoWriter_fourcc('M','J','P','G'), 
            frame_rate,
            (frame_width, frame_height),
        )

        if not video_writer.isOpened():
            print(f"Error: Unable to open video file for writing {output_file_path}.")
            return
        print("Video writer object created.")
    
    print("Detecting objects in the video...")
    total_frames = 0
    try:
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            total_frames+=1
            if not ret:
                break
            start_time = time.time()
            with autocast():
                result = yolo(frame)
            
            for object in result[0]:
                x1, y1, x2, y2 = object.boxes.xyxy.cpu().squeeze()
                confidence = object.boxes.conf.item()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 

                cropped_image = frame[y1:y2,x1:x2]
                text = run_recognition(cropped_image,ocr)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"{text} {confidence:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

            # we duplicate some frames : the input is 30 fps and we are not able to process that fast
            processing_time = time.time() - start_time
            frames_to_duplicate = max(1, int(processing_time / frame_interval))
                
            if output_path is None:
                cv2.imshow('Video', frame)
                cv2.waitKey(1) 
            else:
                for _ in range(frames_to_duplicate):
                    video_writer.write(frame)

    except KeyboardInterrupt: # Ctrl+C stops the processing
        print("Interrupted! Ending ...")
    end = time.time()

    print()
    print("Processing time:", "{:.3f}".format(end-start), "s")
    print("Processed frames:", total_frames)
    print("Processing speed:", "{:.3f}".format(total_frames/(end-start)), "Fps")


    video_capture.release()
    if output_path is not None:
        video_writer.release()
    print("Detection done.")

    if profile:
        profiler.disable()
        with open("output/profiling_results.txt", "w") as f:
            stats = pstats.Stats(profiler, stream=f)
            stats.sort_stats("time")
            stats.print_stats(20)
            stats = pstats.Stats(profiler, stream=f)
            stats.sort_stats("cumulative")
            stats.print_stats(20)
            stats = pstats.Stats(profiler, stream=f)
            stats.sort_stats("calls")
            stats.print_stats(20)

def run_recognition(image,ocr):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
    gaussian_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
    
    process_path = './output/preprocessed_license_plate.jpg'
    cv2.imwrite(process_path, gaussian_image)

    result = ocr.ocr(process_path, rec=True, det=False, cls=True)
    recognized_text = ""
    if result and result[0]:
        for line in result[0]:
            add_text = re.sub(r'[^a-zA-Z0-9]', '', line[0])
            conf = line[1]
            if len(add_text) <=7 and conf > 0.8:
                recognized_text += add_text 

    return recognized_text

def run_detection_image(
    yolo: YOLO,
    ocr,
    image_path: str,
    output_path: str,
) -> None:
    print("Detecting objects in the image...")
    image = cv2.imread(image_path)
    result = yolo(image_path)
    
    for object in result[0]:
        x1, y1, x2, y2 = object.boxes.xyxy.cpu().squeeze()
        confidence = object.boxes.conf.item()

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 

        cropped_image = image[y1:y2,x1:x2]
        text = run_recognition(cropped_image,ocr)

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            image,
            f"{text} {confidence:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )
    if output_path is not None:
        output_file_path = os.path.join(output_path, "output.png")
        cv2.imwrite(output_file_path, image)
    else:
        cv2.imshow("Image", image)
        cv2.waitKey(0)
    print("Detection done.")

def main():
    args = parse_arguments()

    print("Loading YOLO model...")
    yolo = YOLO(args.yolo)
    print("Model loaded.")

    print("Loading OCR model...")
    ocr = PaddleOCR(use_gpu=False,lang="en",show_log=False, use_angle_cls=True)
    print("Model loaded.")

    profile = args.profiler

    if args.input == None or args.input.split(".")[-1] == "mp4" :
        run_detection_video(
            yolo=yolo,
            ocr=ocr,
            video_path=args.input,
            output_path=args.output,
            profile=profile
        )
    else:
        run_detection_image(
            yolo=yolo,
            ocr=ocr,
            image_path=args.input,
            output_path=args.output
        )

if __name__ == "__main__":
    main()
