import argparse
import logging
import sys
import time
import math
import cv2
import numpy as np
import pyopenpose as op
import pickle

def posture_square(poseKeypoints,accuracy_threshold):
    length=np.size(poseKeypoints,0)
    width=np.size(poseKeypoints,1)
    x_max=-10000
    x_min=10000
    y_max=-10000
    y_min=10000
    detected_corrected_rate=0
    not_zero_components=length
    for i in range(length):
        if poseKeypoints[i,2]==0:
            not_zero_components=not_zero_components-1
            continue
        detected_corrected_rate=detected_corrected_rate+poseKeypoints[i,2]
        if poseKeypoints[i,width-1]>=accuracy_threshold: # Only counts in guaranteed points
            if poseKeypoints[i,0]>=x_max:
                x_max=poseKeypoints[i,0]
            if poseKeypoints[i,0]<=x_min:
                x_min=poseKeypoints[i,0]
            if poseKeypoints[i,1]>=y_max:
                y_max=poseKeypoints[i,1]
            if poseKeypoints[i,1]<=y_min:
                y_min=poseKeypoints[i,1]
    x_range=x_max-x_min
    y_range=y_max-y_min
    average_detected_corrected_rate=detected_corrected_rate/not_zero_components
    return (x_range,y_range,average_detected_corrected_rate)


def fifo_append(user_array,value):
    length=np.size(user_array,1)
    full_or_not=False     # Whether the fifo_buff is full or not
    insert_head_velocity=0.0
    average_head_velocity=0.0
    if value[0,2]<0.3:    # if the head keypoint is not detected, or strangely shift. Do not do anything to the user_array, just return the last results Nov_24
        if user_array[0,0]!=0 and user_array[0,1]!=0:
            full_or_not=True
            insert_head_velocity=(user_array[1,length-1]-user_array[1,length-2]) # ATTEMPT 1: Only calculate y-axis velocity
            average_head_velocity=(user_array[1,length-1]-user_array[1,0])/7
        return [user_array, full_or_not, insert_head_velocity, average_head_velocity] # Do not insert a not sure point into the array
    for i in range(length-1):
        user_array[0,i]=user_array[0,i+1]
        user_array[1,i]=user_array[1,i+1]
    user_array[0,length-1]=value[0,0]
    user_array[1,length-1]=value[0,1]
    if user_array[0,0]!=0 and user_array[0,1]!=0: # At this stage the fifo_buff is full
        insert_head_velocity=(user_array[1,length-1]-user_array[1,length-2]) # ATTEMPT 1: Only calculate y-axis velocity
        average_head_velocity=(user_array[1,length-1]-user_array[1,0])/7
        full_or_not=True
    # print(head_monitor)
    # print(full_or_not)
    # print(insert_head_velocity)
    # print(average_head_velocity)

    return [user_array, full_or_not, insert_head_velocity, average_head_velocity]

def body_tilt_detection(human_keypoints):
    
    center_line_angel=math.pi/2 # default 90 degree
    points_are_accurate=True

    head_position=np.zeros([1,3],dtype=float)
    hip_center_position=np.zeros([1,3],dtype=float)
    for i in range(np.size(head_position,1)):
        head_position[0,i]=human_keypoints[0,i]   # Position and accuracy of head point
        hip_center_position[0,i]=human_keypoints[8,i]   # Position and accuracy of hip center point
    if head_position[0,2]<0.2 or hip_center_position[0,2]<0.2:
        # We deem that these two points are not reliable
        points_are_accurate=False
        center_line_angel=math.pi/2
        return [center_line_angel,points_are_accurate] 
    y_distance=abs(head_position[0,1]-hip_center_position[0,1]) # y_distance
    x_distance=abs(head_position[0,0]-hip_center_position[0,0]) # x_distance
    if x_distance<10:
        x_distance=10
    center_line_angel=math.atan(y_distance/x_distance)
    return [center_line_angel, points_are_accurate]

logger = logging.getLogger('TfPoseEstimatorRun')
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


if __name__ == '__main__':
    fps_time = 0
    # Create a fifo buffer to store head position
    head_keypoint=np.zeros([1,3],dtype=float)
    head_history=np.zeros([2,7],dtype=float) # Since the position of a dot is represented by x and y

    params = dict()
    params["model_folder"] = "../../models/"
    params["net_resolution"] = "960x720" 

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()


    print("OpenPose start")
    cap = cv2.VideoCapture('/home/ad/openpose_v2_with_api/examples/tutorial_api_python/video_collection/output_new4.avi')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    #fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # Set the format of outputed video
    #fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G') # Set the format of outputed video
    #out_video = cv2.VideoWriter('/home/ad/openpose_v2_with_api/examples/tutorial_api_python/video_collection/analyze.avi', fourcc, cap.get(cv2.CAP_PROP_FPS), (960,720))
    frame_width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height =int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT))
    print(frame_height)
    print(frame_width)
    #out_video = cv2.VideoWriter('/home/ad/openpose_v2_with_api/examples/tutorial_api_python/video_collection/outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), cap.get(cv2.CAP_PROP_FPS), (frame_height,frame_width))
    
    # STORE RESULT VIDEO NOV_24
    pathOut = '/home/ad/openpose_v2_with_api/examples/tutorial_api_python/video_collection/experiment_results/output_new4.avi'
    fps = 2
    size = (960,720)
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size) # If you do not want to save the test result, comment all the "out" object




    # HEAD monitor
    head_monitor=np.zeros([2,7],dtype=float)
    # velocity threshold for fall detection!!!
    ave_velocity_threshold=20 #12
    # BODY TILT settings
    fall_angle_threshold=math.pi/6

    # FALL HAPPENS!
    fall_happen=False

    # Record the falling period (picture xxxx - xxxx is falling)
    picture_id=0
    # Record the picture_id when fall happens
    fall_happen_picture_id=np.array([],dtype=int)
    append_copy=np.zeros([1,1],dtype=int)

    while (cap.isOpened()):

        #ret_val, dst_before_rotae = cap.read()
        ret_val, dst = cap.read()
        #dst=cv2.rotate(dst_before_rotae,cv2.ROTATE_180)
        dst = cv2.resize(dst,(960,720),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
        if ret_val == False:
            print("Camera read Error")
            break
        picture_id+=1
        print('current picture is ', picture_id)
        datum = op.Datum()
        datum.cvInputData = dst             # Input the captured data into the openpose datum
        opWrapper.emplaceAndPop([datum])    # Analyzing Datum
        fps = 1.0 / (time.time() - fps_time)# After succsessful analyze, calculate the fps
        fps_time = time.time()
        newImage = datum.cvOutputData[:, :, :] # This is the analyze result from openpose
        cv2.putText(newImage , "FPS: %f" % (fps), (20, 40),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)   # Printout the just calculated FPS information
        if fall_happen==True:
             cv2.putText(newImage,"FALL HAPPENS",(360, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
        
        if np.size(datum.poseKeypoints)!=1:
            # Print out the total number of detected person
            human_count = 0
            # remove noise posture in detection
            accuracy_threshold=0.3 
            x_range_threshold=100  
            y_range_threshold=100  
            average_accuracy_rate=0
            human_id_in_keypoints=1e6



            font = cv2.FONT_HERSHEY_SIMPLEX
            for i in range(np.size(datum.poseKeypoints,0)):
                for j in range(25): # first target: print keypoints in picture
                    cv2.putText(newImage,str(j),  ( int(datum.poseKeypoints[i][j][0]) + 10,  int(datum.poseKeypoints[i][j][1])), font, 0.5, (0,255,0), 2) 
                    #print(datum.poseKeypoints[i])
                # second target: remove noise posture in detecting human
                x_range,y_range, average_accuracy_rate=posture_square(datum.poseKeypoints[i],accuracy_threshold)
                
                if x_range<=x_range_threshold and y_range<=y_range_threshold: # IT PROVES THAT IT IS NOT!!! HUMAN
                    continue
                if average_accuracy_rate>=0.3:
                    human_id_in_keypoints=i
                    human_count+=1
            cv2.putText(newImage,"Person number: %i" % (human_count),(20, 680), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0),2)

            if human_id_in_keypoints!= 1e6: # If human_id_in_keypoints is no longer 1e6, it proves that there is a human in picture
                human_keypoints=datum.poseKeypoints[human_id_in_keypoints]
                #print(human_keypoints)

                # HEAD monitor
                [head_monitor, full_or_not, insert_head_velocity, average_head_velocity]=fifo_append(head_monitor,human_keypoints)
                print(human_keypoints)
                # print(head_monitor)
                # print(full_or_not)
                # print(insert_head_velocity)
                # print(average_head_velocity)
                if full_or_not == True:         # The HEAD monitor array is full now, its results are reliable
                    #print(insert_head_velocity)
                    #print(average_head_velocity)
                    cv2.putText(newImage,"insert v: %f" % (insert_head_velocity),(600, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0),2)
                    cv2.putText(newImage,"average v: %f" % (average_head_velocity),(600, 680), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0),2)

                    # If the average head velocity is beyond the threshold, the fall detcion algorithm should start working
                    if average_head_velocity>=ave_velocity_threshold:
                        # BODY TILT detection
                        center_line_angel, points_are_accurate=body_tilt_detection(human_keypoints)

                        if points_are_accurate==True:
                            cv2.putText(newImage,"tilt_angle: %f" % (center_line_angel),(600, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0),2)
                            if center_line_angel<=math.pi/5: 
                                fall_happen=True
                                append_copy[0]=picture_id
                                fall_happen_picture_id=np.append(fall_happen_picture_id,append_copy)
                                a_file=open("fall_id.txt","w")
                                np.savetxt(a_file,fall_happen_picture_id)
                                #pickle.dump(fall_happen_picture_id,a_file)
                                a_file.close()







        #out_video.write(newImage)
        cv2.imshow("test",newImage)
        out.write(newImage)
        cv2.waitKey(1)
        #out_video.write(newImage)


    print('Fall happen picture id: ',fall_happen_picture_id)
    cv2.destroyAllWindows()
    #out_video.release()
    cap.release()
    out.release()