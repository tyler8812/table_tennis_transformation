import numpy as np
import cv2 as cv
from skimage.transform import ProjectiveTransform
import matplotlib.pyplot as plt

width = 822
height = 457

four_point = []

result = cv.imread('black.png')
img = cv.imread('black.png')

cv.imshow('image', img)
data = np.asarray([])

t = ProjectiveTransform()

def click_event(event, x, y, flags, params):
    
    global img, four_point, t
    if len(four_point) == 4:
        cv.line(img, (four_point[0][0], four_point[0][1]), (four_point[1][0], four_point[1][1]), (255, 255, 255), 3)
        cv.line(img, (four_point[1][0], four_point[1][1]), (four_point[2][0], four_point[2][1]), (255, 255, 255), 3)
        cv.line(img, (four_point[2][0], four_point[2][1]), (four_point[3][0], four_point[3][1]), (255, 255, 255), 3)
        cv.line(img, (four_point[3][0], four_point[3][1]), (four_point[0][0], four_point[0][1]), (255, 255, 255), 3)
        src = np.asarray(
            [[four_point[0][0], four_point[0][1]], [four_point[1][0], four_point[1][1]], [four_point[2][0], four_point[2][1]], [four_point[3][0], four_point[3][1]]])
        dst = np.asarray([[300, 100], [300, 100 + height], [300 + width, 100 + height], [300 + width, 100]])
        if not t.estimate(src, dst): raise Exception("estimate failed")
        cv.line(result, (dst[0][0], dst[0][1]), (dst[1][0], dst[1][1]), (255, 255, 255), 3)
        cv.line(result, (dst[1][0], dst[1][1]), (dst[2][0], dst[2][1]), (255, 255, 255), 3)
        cv.line(result, (dst[2][0], dst[2][1]), (dst[3][0], dst[3][1]), (255, 255, 255), 3)
        cv.line(result, (dst[3][0], dst[3][1]), (dst[0][0], dst[0][1]), (255, 255, 255), 3)
    if event == cv.EVENT_LBUTTONDOWN:
        print(x, ' ', y)
        if len(four_point) < 4:
            four_point.append([x, y])
        else:
            data_local = t([x, y])
            cv.circle(result, (int(data_local[0][0]), int(data_local[0][1])), 3, (255, 231, 0), -1)
        cv.circle(img, (x, y), 3, (255, 255, 255), -1)  

    cv.imshow('image', img) 
    cv.imshow('result', result)   
cv.setMouseCallback('image', click_event)
cv.waitKey(0)








# cv.circle(img, (x, y), 3, (255, 255, 255), -1)
result_img1 = cv.imread('black.png')
cv.line(result_img1, (four_point[0][0], four_point[0][1]), (four_point[1][0], four_point[1][1]), (255, 255, 255), 3)
cv.line(result_img1, (four_point[1][0], four_point[1][1]), (four_point[2][0], four_point[2][1]), (255, 255, 255), 3)
cv.line(result_img1, (four_point[2][0], four_point[2][1]), (four_point[3][0], four_point[3][1]), (255, 255, 255), 3)
cv.line(result_img1, (four_point[3][0], four_point[3][1]), (four_point[0][0], four_point[0][1]), (255, 255, 255), 3)
for i in range(len(data)):
    cv.circle(result_img1, (data[i][0], data[i][1]), 3, (255, 231, 0), -1)


result_img2 = cv.imread('black.png')
cv.line(result_img2, (dst[0][0], dst[0][1]), (dst[1][0], dst[1][1]), (255, 255, 255), 3)
cv.line(result_img2, (dst[1][0], dst[1][1]), (dst[2][0], dst[2][1]), (255, 255, 255), 3)
cv.line(result_img2, (dst[2][0], dst[2][1]), (dst[3][0], dst[3][1]), (255, 255, 255), 3)
cv.line(result_img2, (dst[3][0], dst[3][1]), (dst[0][0], dst[0][1]), (255, 255, 255), 3)
for i in range(len(data)):
    cv.circle(result_img2, (int(data_local[i][0]), int(data_local[i][1])), 3, (255, 231, 0), -1)

cv.imshow('image1', result_img1)
cv.imshow('image2', result_img2)
cv.waitKey(0)


# plt.figure()
# plt.plot(src[[0,1,2,3,0], 0], src[[0,1,2,3,0], 1], '-')
# plt.plot(data.T[0], data.T[1], 'o')
# plt.figure()
# plt.plot(dst.T[0], dst.T[1], '-')
# plt.plot(data_local.T[0], data_local.T[1], 'o')
# plt.show()
