import numpy as np
import cv2 as cv
from skimage.transform import ProjectiveTransform
import matplotlib.pyplot as plt

width = 822
height = 457

four_point = []

result = cv.imread('black.png')
img = cv.imread('black.png')
dst = np.asarray([])

cv.imshow('image', img)

line_for_correct = []

t = ProjectiveTransform()

def click_event(event, x, y, flags, params):
    
    global img, four_point, t, dst

    if len(four_point) == 4:
        cv.line(img, (four_point[0][0], four_point[0][1]), (four_point[1][0], four_point[1][1]), (255, 255, 255), 3)
        cv.line(img, (four_point[1][0], four_point[1][1]), (four_point[2][0], four_point[2][1]), (255, 255, 255), 3)
        cv.line(img, (four_point[2][0], four_point[2][1]), (four_point[3][0], four_point[3][1]), (255, 255, 255), 3)
        cv.line(img, (four_point[3][0], four_point[3][1]), (four_point[0][0], four_point[0][1]), (255, 255, 255), 3)
        
        for i in range(0, 3):
            point1 = ((np.array([four_point[i+1][0], four_point[i+1][1]]) - np.array([four_point[i][0], four_point[i][1]])) / 3 + np.array([four_point[i][0], four_point[i][1]])).astype(int)
            point2 = ((np.array([four_point[i+1][0], four_point[i+1][1]]) - np.array([four_point[i][0], four_point[i][1]])) / 3 * 2 + np.array([four_point[i][0], four_point[i][1]])).astype(int)
            line_for_correct.append(point1)
            line_for_correct.append(point2)

        point1 = ((np.array([four_point[0][0], four_point[0][1]]) - np.array([four_point[3][0], four_point[3][1]])) / 3 + np.array([four_point[3][0], four_point[3][1]])).astype(int)
        point2 = ((np.array([four_point[0][0], four_point[0][1]]) - np.array([four_point[3][0], four_point[3][1]])) / 3 * 2 + np.array([four_point[3][0], four_point[3][1]])).astype(int)
        line_for_correct.append(point1)
        line_for_correct.append(point2)

        cv.line(img, (line_for_correct[0][0], line_for_correct[0][1]), (line_for_correct[5][0], line_for_correct[5][1]), (255, 0, 0), 3)
        cv.line(img, (line_for_correct[1][0], line_for_correct[1][1]), (line_for_correct[4][0], line_for_correct[4][1]), (255, 0, 0), 3)
        cv.line(img, (line_for_correct[2][0], line_for_correct[2][1]), (line_for_correct[7][0], line_for_correct[7][1]), (255, 0, 0), 3)
        cv.line(img, (line_for_correct[3][0], line_for_correct[3][1]), (line_for_correct[6][0], line_for_correct[6][1]), (255, 0, 0), 3)
        

        src = np.asarray(
            [[four_point[0][0], four_point[0][1]], [four_point[1][0], four_point[1][1]], [four_point[2][0], four_point[2][1]], [four_point[3][0], four_point[3][1]]])
        dst = np.asarray([[300, 100], [300, 100 + height], [300 + width, 100 + height], [300 + width, 100]])

        if not t.estimate(src, dst): raise Exception("estimate failed")

        cv.line(result, (dst[0][0], dst[0][1]), (dst[1][0], dst[1][1]), (255, 255, 255), 3)
        cv.line(result, (dst[1][0], dst[1][1]), (dst[2][0], dst[2][1]), (255, 255, 255), 3)
        cv.line(result, (dst[2][0], dst[2][1]), (dst[3][0], dst[3][1]), (255, 255, 255), 3)
        cv.line(result, (dst[3][0], dst[3][1]), (dst[0][0], dst[0][1]), (255, 255, 255), 3)

        corrected_line = t(line_for_correct).astype(int)
        cv.line(result, (corrected_line[0][0], corrected_line[0][1]), (corrected_line[5][0], corrected_line[5][1]), (255, 0, 0), 3)
        cv.line(result, (corrected_line[1][0], corrected_line[1][1]), (corrected_line[4][0], corrected_line[4][1]), (255, 0, 0), 3)
        cv.line(result, (corrected_line[2][0], corrected_line[2][1]), (corrected_line[7][0], corrected_line[7][1]), (255, 0, 0), 3)
        cv.line(result, (corrected_line[3][0], corrected_line[3][1]), (corrected_line[6][0], corrected_line[6][1]), (255, 0, 0), 3)

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
print(dst)
print(t.inverse(dst))
