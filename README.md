# YOLO TRAIN GUI 
A GUI Tool for YOLO Training  

## Background
Trying to build a GUI tool for YOLO Training so that it wolud be easy to use.  
Users shall focus more on parameters other than scripts.  

## Requirements
### You shall have YOLO environment ready
```bash
pip install ultralytics
```
### Also Pyqt6 shall be installed. 
```bash
pip install pyqt6
```
I've tried to pack as an app for mac or an exe for windows.  
But it appears that there's limitations, especially with virtual environment.  
I would try later, so a .py file would be released first.  
### You shall get .pt models dataset ready
especially .yaml file, the .yaml for data set shall with the /train, /val /test folders. and looks like:
```yaml
train: ../train/images
val: ../valid/images
test: ../test/images
```
## How to Use
In the environment you've got YOLO and pyqt6 ready, run the YOLO_Train_GUI.py.  
You shall see the following screen:  
![ScreenShot](/img/screen_shot.png)
You could adjust all the parameters displayed.  

## Special
If you'd like to try a self-adjusted YOLO model, you shall prepare a personalized yaml.  
For example, you'd like to insert a CBAM module in.  

## Remarks
I've only tested on Apple Mac Mini M4 with anaconda environment and all default parameters only.  
Wish this tool would help you.  
I would try to update the tool and make it more functional.  
If there's issue, please try to modify the source code.  
You could submit an issue, but slow response would be expected.   
Thank you for your understanding.  

## About Me
If you are interested, you could find my [Blog Here](https://kevinblog.de5.net "Kevin's Blog")
Thank you.  
