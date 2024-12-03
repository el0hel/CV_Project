# Recognition on one image
```python
python src/main.py -i .\input\image.png  -y best.pt -o .\output\
```

# Recognition on a video and output to a folder
```python
python src/main.py -i .\input\video.mp4 -o output\ -y best.pt
```
# Recognition on a video and output to the screen directly without saving to a folder
```python
python src/main.py -i .\input\video.mp4 -y best.pt
```