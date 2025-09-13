import cv2
import numpy as np


def cartoonify_image(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray=cv2.medianBlur(gray,5)
    edges=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,9,9)
    color=cv2.bilateralFilter(img,9,300,300)
    cartoon=cv2.bitwise_and(color,color,mask=edges)
    return cartoon


def apply_filter(frame,filter_name):
    if filter_name=="gray":
        return cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    elif filter_name=="blur":
        return cv2.GaussianBlur(frame,(15,15),0)
    
    elif filter_name=="Cartoonify":
        frame=cartoonify_image(frame)
    
    elif filter_name=="sepia":
        sepia_filter = np.array([[0.272, 0.534, 0.131],
                                 [0.349, 0.686, 0.168],
                                 [0.393, 0.769, 0.189]])
        return cv2.transform(frame, sepia_filter)

    else:
        return frame