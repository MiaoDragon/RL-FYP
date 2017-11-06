import cv2
img = cv2.imread('Lenna.png')
img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
y, _, _ = cv2.split(img_yuv)

# scale
res = cv2.resize(y, (84,84), interpolation = cv2.INTER_CUBIC)
cv2.imshow('lenna1', res)
print(res)
print(res.shape)
#y = cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)
#cv2.imshow('lenna2', y)
cv2.waitKey(0)
cv2.destroyAllWindows()
