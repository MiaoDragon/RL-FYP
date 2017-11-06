import cv2
import numpy as np
class Preprocessor:
    def __init__(self):
        self.me = 0
        self.opponent = 1
        self.prev_value = [255.,255.]  # previous values
        self.prev_state = {'switch': False} # previous state
        self.counter = 0 # when first switching, then wait the number of frames

    def switch(self):
        self.me = 1 - self.me
        self.opponent = 1 - self.opponent

    def preprocess(self, img):
        # TODO: add current hp info
        # return: H x W
        original_image = img
        # convert to gray
        processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # edge detection
        processed_img = cv2.Canny(processed_img, threshold1=200, threshold2=300)
        return processed_img


    def reward_cal(self, img):
        # this is specific to this task
        DIE_RWD = 2.0
        HIT_RWD = 1.0
        SWT_COUNT = 20 # skip # frames after switch
        #P1
        #cv2.imshow('window2',img[88:90,710:722])
        #P2
        #cv2.imshow('window3',img[88:90,760:772])
        #img1 = img[88:90,710:722]
        #img2 = img[88:90, 760:772]
        img1 = img[88:90,716:722]
        img2 = img[88:90, 766:772]  # hp bar
        #img3 = img[60:70,705:717]  # face capture
        #img4 = img[60:70, 760:772]
        #img3 = img[82:86,697:704]
        #img4 = img[82:86, 745:752]  # life counter
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
        # skip # frames after switching
        if self.counter != 0:
            print(self.counter)
            self.counter = (self.counter + 1) % SWT_COUNT
            self.prev_value = [r1, r2]
            return 0.
        epslon = 0.1 * 255.0
        diff_flag = False
        for i in y1.flat:
            if abs(i - r1) > epslon:
                # p1 value different, then should be switching
                diff_flag = True
                break

        if diff_flag and not self.prev_state['switch']:
            print('switching...')
            self.switch()
            self.prev_value = [r1, r2]
            self.prev_state['switch'] = True
            self.counter = 1
            return 0.  # reward is 0
        if not diff_flag:
            self.prev_state['switch'] = False
            # 2. Die: one player change to larger value, and
            if r1 > self.prev_value[0] and r1 == 255.0:
                # P1 die
                self.prev_value = [r1, r2]
                return (2*self.me-1) * DIE_RWD
            if r2 > self.prev_value[1] and r2 == 255.0:
                # P2 die
                self.prev_value = [r1, r2]
                return (2*self.opponent-1) * DIE_RWD

            # 3. hit: one player change to smaller value
            if r1 < self.prev_value[0]:
                # P1 hitten
                self.prev_value = [r1, r2]
                #self.prev_life = [l1, l2]
                return (2*self.me-1) * HIT_RWD
            if r2 < self.prev_value[1]:
                # P2 hitten
                self.prev_value = [r1, r2]
                return (2*self.opponent-1) * HIT_RWD

            self.prev_value = [r1, r2]
            return 0.
        # 5. In the middle of switching
        self.prev_value = [r1, r2]
        return 0.
