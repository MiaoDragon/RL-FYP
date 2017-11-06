import numpy as np
from PIL import Image, ImageGrab
from mss import mss
import cv2
import time
import Quartz.CoreGraphics as CG
import struct
import keyboard
import torch
#from Preprocessor import Preprocessor
def roi(img, vertices):
    #blank mask:
    mask = np.zeros_like(img)
    # fill the mask
    cv2.fillConvexPoly(mask, vertices, 255)
    # now only show the area that is the mask
    masked = img + cv2.bitwise_and(img, mask)
    return masked

def process_img(image):
    original_image = image
    # convert to gray
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # edge detection
    processed_img = cv2.Canny(processed_img, threshold1=200, threshold2=300)
    #print(type(processed_img))
    processed_img = torch.from_numpy(processed_img).type(torch.FloatTensor) / 255.0
    return processed_img

def screen_record_v1():
    last_time = time.time()
    while(True):
        # 800x600 windowed mode
        printscreen = np.array(ImageGrab.grab(bbox=(100,100,640,640)))
        #printscreen_numpy = np.array(printscreen_pil,dtype='uint8')\
        #                    .reshape(printscreen_pil.size[1],printscreen_pil.size[0],4)
        print('loop took {0} seconds' . format(time.time()-last_time))
        last_time = time.time()
        new_screen = process_img(printscreen)
        cv2.imshow('window',new_screen)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

def screen_record_v2():
    mon = {'top': 100, 'left': 100, 'width': 800, 'height':640}
    sct = mss()
    last_time = time.time()
    while 1:
        img = np.array(sct.grab(mon))
        #cv2.imshow('test',img)
        #img = Image.frombytes('RGB', sct_img.size,sct_img.rgb)
        #cv2.imshow('test', np.array(img))

        print('loop took {0} seconds' . format(time.time()-last_time))
        last_time = time.time()

        #if cv2.waitKey(25) & 0xFF == ord('q'):
        #    cv2.destroyAllWindows()
        #    break
def screen_record_v3():
    last_time = time.time()
    while 1:
        region = CG.CGRectMake(100, 100, 400, 400)
        image = CG.CGWindowListCreateImage(region, CG.kCGWindowListOptionOnScreenOnly, CG.kCGNullWindowID, CG.kCGWindowImageDefault)
        width = CG.CGImageGetWidth(image)
        height = CG.CGImageGetHeight(image)
        prov = CG.CGImageGetDataProvider(image)
        data = CG.CGDataProviderCopyData(prov)
        byteperrow = CG.CGImageGetBytesPerRow(image)
        #print prov
        #<CGDataProvider 0x7fc19b1022f0>
        #print type(data)
        #<objective-c class __NSCFData at 0x7fff78073cf8>
        img = np.frombuffer(data, dtype=np.uint8)
        img = img.reshape((height, byteperrow//4,4))
        img = img[:, :width, :]
        print('loop took {0} seconds' . format(time.time()-last_time))
        img = process_img(img)
        # the above take roughly 0.01s
        last_time = time.time()
        cv2.imshow('window',img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        # cv2 will increase 0.09s

def screen_record():
    region = CG.CGRectMake(100, 122, 400, 378)
    #image = CG.CGWindowListCreateImage(region, CG.kCGWindowListOptionOnScreenOnly, CG.kCGNullWindowID, CG.kCGWindowImageDefault)
    image = CG.CGWindowListCreateImage(region, CG.kCGWindowListOptionOnScreenOnly, CG.kCGNullWindowID, CG.kCGWindowImageNominalResolution)
    width = CG.CGImageGetWidth(image)
    height = CG.CGImageGetHeight(image)
    prov = CG.CGImageGetDataProvider(image)
    data = CG.CGDataProviderCopyData(prov)
    byteperrow = CG.CGImageGetBytesPerRow(image)
    img = np.frombuffer(data, dtype=np.uint8)
    img = img.reshape((height, byteperrow//4,4))
    img = img[:, :width, :]
    return img

def life_dec(img):
    img1 = img[82:86,697:704]
    img2 = img[82:86, 745:752]
    cv2.imshow('window2', img1)
    cv2.imshow('window3', img2)
    img1_yuv = cv2.cvtColor(img1, cv2.COLOR_BGR2YUV)
    y1, _, _ = cv2.split(img1_yuv)
    img2_yuv = cv2.cvtColor(img2, cv2.COLOR_BGR2YUV)
    y2, _, _ = cv2.split(img2_yuv)

def face_cap(img):
    img1 = img[60:70,705:717]
    img2 = img[60:70, 760:772]
    cv2.imshow('window2', img1)
    cv2.imshow('window3', img2)

def reward_cal(img):
    # this is specific to this task

    #P1
    #cv2.imshow('window2',img[88:90,710:722])
    #P2
    #cv2.imshow('window3',img[88:90,760:772])
    #img1 = img[88:90,710:722]
    #img2 = img[88:90, 760:772]
    img1 = img[88:90,716:722]
    img2 = img[88:90, 766:772]
    img1_yuv = cv2.cvtColor(img1, cv2.COLOR_BGR2YUV)
    y1, _, _ = cv2.split(img1_yuv)
    img2_yuv = cv2.cvtColor(img2, cv2.COLOR_BGR2YUV)
    y2, _, _ = cv2.split(img2_yuv)
    # for test
    #yout1 = cv2.cvtColor(y1, cv2.COLOR_GRAY2BGR)
    #yout2 = cv2.cvtColor(y2, cv2.COLOR_GRAY2BGR)
    #cv2.imshow('window2', yout1)
    #cv2.imshow('window3', yout2)
    r1 = np.mean(y1)
    r2 = np.mean(y2)
    epslon = 0.3 * 255.0
    for i in y1.flat:
        if abs(i - r1) > epslon:
            print('p1 value different!')
            break
    for j in y2.flat:
        if abs(j - r2) > epslon:
            print('p2 value different!')
            break
    # 255: white
    # smaller value: darker

    # Retrieve Information about Reward:
    # 1. Switch Player: two player value different
    # 2. Die: one player change to larger value
    # 3. hit: oen player change to smaller value

#----------------------------
# test
#screen_record_v1()
#screen_record_v2()
#screen_record_v3()
#Prep = Preprocessor()
for i in list(range(4))[::-1]:
    print(i+1)
    time.sleep(1)

while 1:
    # control
    #keyboard.press('d')
    # show
    img = screen_record()
    #print(np.array(img).shape)
    #vertices = np.array([[712, 88], [768, 85],[768, 100],[712, 100]], np.int32)
    # preprocess
    #img = process_img(img)
    cv2.imshow('window',img)
    print(process_img(img))
    #time.sleep(0.2)
    #life_dec(img)
    #face_cap(img)
    #reward = Prep.reward_cal(img)
    #if reward != 0.0:
    #    print('reward: {}' . format(reward))
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
