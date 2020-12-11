import argparse
import logging
import sys
import time
import math
import cv2
import numpy as np

if __name__ == '__main__':

    print("OpenPose start")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ret_val, img = cap.read()
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # Set the format of outputed video
    out_video = cv2.VideoWriter('/home/ad/openpose_v2_with_api/examples/tutorial_api_python/video_collection/capture1.avi', fourcc, cap.get(cv2.CAP_PROP_FPS), (1280,720))

    count = 0

    if cap is None:
        print("Camera Open Error")
        sys.exit(0)
    while cap.isOpened () and count<=100:
        ret_val, dst = cap.read()
        if ret_val == False:
            print("Camera read Error")
            break
        cv2.imshow("test",dst)
        cv2.waitKey(1)
        out_video.write(dst)
        count += 1

    cv2.destroyAllWindows()
    out_video.release()
    cap.release()

    