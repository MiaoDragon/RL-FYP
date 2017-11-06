import numpy as np
from PIL import Image, ImageGrab
from mss import mss
import cv2
import time
import Quartz.CoreGraphics as CG
import struct
import keyboard
def wait(T):
    for i in list(range(T))[::-1]:
        print(i+1)
        time.sleep(1)

class Sensor:
    def __init__(self,x=100,y=100,width=400,height=400):
        self.region = CG.CGRectMake(x, y, width, height)
        self.width = width
        self.height = height
        self.x = x
        self.y = y
    def screen_record(self):
        image = CG.CGWindowListCreateImage(self.region, CG.kCGWindowListOptionOnScreenOnly,
                                        CG.kCGNullWindowID, CG.kCGWindowImageDefault)
        width = CG.CGImageGetWidth(image)
        height = CG.CGImageGetHeight(image)
        prov = CG.CGImageGetDataProvider(image)
        data = CG.CGDataProviderCopyData(prov)
        byteperrow = CG.CGImageGetBytesPerRow(image)
        img = np.frombuffer(data, dtype=np.uint8)
        img = img.reshape((height, byteperrow//4,4))
        img = img[:, :width, :]
        return img

"""
wait(4)
while 1:
    # control
    #keyboard.press('d')
    # show
    sensor = Sensor()
    img = sensor.screen_record()
    #vertices = np.array([[712, 88], [768, 85],[768, 100],[712, 100]], np.int32)
    # preprocess
    #img = process_img(img)
    #cv2.imshow('window',img)
    #time.sleep(0.2)
    #life_dec(img)
    #face_cap(img)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
"""
