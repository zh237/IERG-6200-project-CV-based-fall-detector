
# ###########################################################################################
# # Method 1
# import cv2
# import os
# def getFrame(sec):
#     vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
#     hasFrames,image = vidcap.read()
#     name = './data/frame' + str(count) + '.jpg'
#     if hasFrames:
#         cv2.imwrite(name, image)     # save frame as JPG file
#     return hasFrames



# try: 

#     # creating a folder named data 
#     if not os.path.exists('data'): 
#         os.makedirs('data') 

# # if not created then raise error 
# except OSError: 
#     print ('Error: Creating directory of data') 

# vidcap = cv2.VideoCapture('/home/ad/openpose_v2_with_api/examples/tutorial_api_python/video_collection/video_photo_transform/v1.avi')
# sec = 0
# frameRate = 1/6.2 #//it will capture image in each 0.5 second
# count=1
# success = getFrame(sec)
# while success:
#     count = count + 1
#     sec = sec + frameRate
#     sec = round(sec, 2)
#     success = getFrame(sec)






######################################################################################################################
#Method 2
# Importing all necessary libraries 
import cv2 
import os 

# Read the video from specified path 
cam = cv2.VideoCapture('/home/ad/openpose_v2_with_api/examples/tutorial_api_python/video_collection/video_photo_transform/v4.avi') 

try: 

    # creating a folder named data 
    if not os.path.exists('data_test_1'): 
        os.makedirs('data_test_1') 

# if not created then raise error 
except OSError: 
    print ('Error: Creating directory of data') 

# frame 
currentframe = 0

while(True): 

    # reading from frame 
    ret,frame = cam.read() 

    if ret: 
        # if video is still left continue creating images 
        name = './data_test_1/frame' + str(currentframe) + '.jpg'
        print ('Creating...' + name) 

        # writing the extracted images 
        cv2.imwrite(name, frame) 

        # increasing counter so that it will 
        # show how many frames are created 
        currentframe += 1
    else: 
        break

# Release all space and windows once done 
cam.release() 
cv2.destroyAllWindows() 