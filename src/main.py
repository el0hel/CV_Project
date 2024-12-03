# coding: utf-8
import os

from ultralytics import YOLO
from argparse import ArgumentParser
import torch
import cv2
import re
import easyocr
import math

def distance(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def parse_arguments():
    parser = ArgumentParser(description="Inference script for detecting and recognising license plates using pre-trained YOLO and OCR model.")

    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the input file.")
    parser.add_argument("-o", "--output", type=str, required=False, help="Path to the output file.")
    parser.add_argument("-y", "--yolo", type=str, required=True, help="Path to the YOLO model.")

    return parser.parse_args()

def run_detection_video(
    yolo: YOLO,
    reader,
    video_path: str,
    output_path: str,
) -> None:
    print("Creating video capture object ...")
    video_capture = cv2.VideoCapture(video_path)

    frame_rate = int(video_capture.get(cv2.CAP_PROP_FPS))
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not video_capture.isOpened():
        print(f"Error: Unable to open video file for reading {video_path}.")
        return
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
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        
        if not ret:
            break
 
        result = yolo(frame)
        
        for object in result[0]:
            x1, y1, x2, y2 = object.boxes.xyxy.cpu().squeeze()
            label = int(object.boxes.cls.item())
            class_name = "license plate" 
            confidence = object.boxes.conf.item()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 

            cropped_image = frame[y1:y2,x1:x2]
            text = run_recognition(cropped_image,reader)

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

            
        if output_path is None:
            cv2.imshow('Video', frame)
            cv2.waitKey(1) # required to enable window content painting
        else:
            video_writer.write(frame)

    video_capture.release()
    video_writer.release()
    print("Detection done.")

def run_recognition(image,reader):
    #scaled_image = cv2.resize(image, (400, 300))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
    gaussian_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
    
    process_path = './output/preprocessed_license_plate.jpg'
    cv2.imwrite(process_path, gaussian_image)

    result = reader.readtext(process_path)

    recognized_text = ""
    max_area = 0
    for (bbox, text, prob) in result:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        width = distance(top_left, top_right)
        height = distance(top_left, bottom_left)    
        area = width * height
        if area > max_area:
            max_area = area
            recognized_text = text
    recognized_text = re.sub('[^a-zA-Z0-9]', '', recognized_text)
    recognized_text.upper()
    return recognized_text

def run_detection_image(
    yolo: YOLO,
    reader,
    image_path: str,
    output_path: str,
) -> None:
    print("Detecting objects in the image...")
    output_file_path = os.path.join(output_path, "output.png")
    image = cv2.imread(image_path)
    result = yolo(image_path)
    
    for object in result[0]:
        x1, y1, x2, y2 = object.boxes.xyxy.cpu().squeeze()
        label = int(object.boxes.cls.item())
        class_name = "license plate"
        confidence = object.boxes.conf.item()

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 

        cropped_image = image[y1:y2,x1:x2]
        text = run_recognition(cropped_image,reader)

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

    cv2.imwrite(output_file_path, image)
    print("Detection done.")

def main():
    args = parse_arguments()

    print("Loading YOLO model...")
    yolo = YOLO(args.yolo)
    print("Model loaded.")
    print(args.output)
    reader = easyocr.Reader(['en'])

    if args.input.split(".")[-1] == "mp4":
        run_detection_video(
            yolo=yolo,
            reader=reader,
            video_path=args.input,
            output_path=args.output,
        )
    else:
        run_detection_image(
            yolo=yolo,
            reader=reader,
            image_path=args.input,
            output_path=args.output,
        )

if __name__ == "__main__":
    main()

