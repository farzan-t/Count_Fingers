


import cv2
import numpy as np
def calc_accum_avg(frame, accumulated_weight):
    global background

    if background is None:
        background = frame.copy().astype('float')
        return None
    cv2.accumulateWeighted(frame, background, accumulated_weight)

def segment(frame, threshold_min=60): #25 default

    diff = cv2.absdiff(background.astype('uint8'), frame)
    ret, thresholded = cv2.threshold(diff, threshold_min, 255, cv2.THRESH_BINARY)
    image, contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None

    else:
        # ASSUMING THE LARGEST EXTERNAL CONTOUR IN ROI, IS THE HAND
        hand_segment = max(contours, key=cv2.contourArea)
        return (thresholded, hand_segment)


def count_fingers(thresholded, hand_segment):
    conv_hull = cv2.convexHull(hand_segment)

    # TOP
    top = tuple(conv_hull[conv_hull[:, :, 1].argmin()][0])
    bottom = tuple(conv_hull[conv_hull[:, :, 1].argmax()][0])
    left = tuple(conv_hull[conv_hull[:, :, 0].argmin()][0])
    right = tuple(conv_hull[conv_hull[:, :, 0].argmax()][0])

    cX = (left[0] + right[0]) // 2
    cY = (top[1] + bottom[1]) // 2

    distance = pairwise.euclidean_distances([[cX, cY]], Y=[left, right, top, bottom])[0]
    max_distance = distance.max()
    radius = int(0.7 * max_distance)
    circumfrence = (2 * np.pi * radius)
    circular_roi = np.zeros_like(thresholded, dtype="uint8")
    cv2.circle(circular_roi, (cX, cY), radius, 255, 10)
    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)
    image, contours, hierarchy = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    count = 0

    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        out_of_wrist = (cY + (cY * 0.25)) > (y + h)
        limit_points = ((circumfrence * 0.25) > cnt.shape[0])
        if out_of_wrist and limit_points:
            count += 1
    return count

from sklearn.metrics import pairwise
background = None

top = 100
bottom = 275
right = 400
left = 575
accumulated_weight = .25






cam = cv2.VideoCapture(0)
num_frames = 0

while True:

    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    frame_copy = frame.copy()
    roi = frame[top:bottom, right:left]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    if num_frames < 60:
        calc_accum_avg(gray, accumulated_weight)
        if num_frames <= 59:
            cv2.putText(frame_copy, 'WAIT. GETTING BACKGROUND', (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Finger Count', frame_copy)
    else:

        hand = segment(gray)
        if hand is not None:
            thresholded, hand_segment = hand

            # DRAWS CONTOURS AROUND REAL HAND IN LIVE STREAM
            # cv2.drawContours(frame_copy, [hand_segment + (roi_right, roi_top)], -1, (255, 0, 0), 1)
            fingers = count_fingers(thresholded, hand_segment)
            cv2.putText(frame_copy, str(fingers), ((left + right) // 2, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # cv2.putText(thresholded, str(fingers), ((50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255 , 2))
            cv2.imshow('Thresholded', thresholded)



    cv2.rectangle(frame_copy, (left, top), (right, bottom), (0, 255, 0), 2)
    num_frames += 1
    cv2.imshow('Finger Count', frame_copy)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cam.release()
cv2.destroyAllWindows()

