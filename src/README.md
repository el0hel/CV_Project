# YOLO and PaddleOCR License Plate Detection and Recognition

This program detects and recognizes license plates from images or videos using a combination of YOLO for object detection and PaddleOCR for text recognition.
For a video, it draws the bounding boxes of the found license plates and displays above each of them the id of the tracked object and its license plate number.
For an image, it draws the bounding boxes of the found license plates and displays above each of them the license plate number and the confidence score of the YOLO model.
If a directory is passed as input, the program will check all valid images and videos inside and do the detection and recognition accordingly.

## Requirements

To install all the necessary modules that can be found in requirements.txt, run :
```cmd
pip install -r requirements.txt
```

## Command line arguments

- ```-i or --input (optional)```: Path to the input image file (.jpg, .png, .jpeg, .bmp), video ifle (.mp4, .mov) or directory. If not provided, the script will use the default system camera.
- ```-o or --output (optional)```: Path to save the processed output (image or video).  If not provided, the results will be displayed on the default screen.
- ```-y or --yolo (required)```: Path to the YOLO model.
- ```-p or --profiler (optional)```: Enable profiling of the program for performance analysis.

## Detecting License Plates in Videos
To process a video file,:
```cmd
python main.py -i <video_path> -o <output_path> -y <yolo_model_path>
python src/main.py -i .\input\video.mp4 -o .\output\ -y best.pt # example of command to run from the parent repository
```
To process the camera input:
```cmd
python main.py -o <output_path> -y <yolo_model_path>
python src/main.py  -o .\output\ -y best.pt # example of command to run from the parent repository
```

## Detecting License Plates in Images
To process an image file:
```cmd
python main.py -i <image_path> -o <output_path> -y <yolo_model_path>
python src/main.py -i .\input\image.png -o .\output\ -y best.pt # example of command to run from the parent repository
```

## Passing Directory
To process a directory file:
```cmd
python main.py -i <directory_path> -o <output_path> -y <yolo_model_path>
python src/main.py -i .\input\ -o .\output\ -y best.pt # example of command to run from the parent repository
```


## Profiling
Profiling results are saved in ```output/profiling_results.txt```, detailing execution time and function calls.
To enable profiling add ```-p``` to the command line.
