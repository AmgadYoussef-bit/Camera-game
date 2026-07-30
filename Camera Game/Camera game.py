import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2 as cv
import sys
import random

network_path = r"D:\Neural networks\hand_landmarker.task"

thumbs_up = cv.imread(r"D:\Pictures\thumbs up.jpeg")

base__options = mp.tasks.BaseOptions
Hand_landmarker = mp.tasks.vision.HandLandmarker
Hand_landmarker_options = mp.tasks.vision.HandLandmarkerOptions
Hand_landmarker_results = mp.tasks.vision.HandLandmarkerResult
Vision_running_mode = mp.tasks.vision.RunningMode

results = "Nothing Detected"
image = None
def call_backfun(result: Hand_landmarker_results , output_image: mp.Image, timestamp_ms: int):
  

    global image,results
    if len(result.hand_landmarks) != 0:
        results = result.hand_landmarks[0]
        # print(dir(result.hand_landmarks[0][1]))
    image = output_image.numpy_view()
    # print("hand landmarker result {}".format(results), end="\n\n")



    

options = Hand_landmarker_options(
    base_options = base__options(model_asset_path=r"D:\Neural networks\hand_landmarker.task"),
    running_mode = Vision_running_mode.LIVE_STREAM,
    result_callback = call_backfun
)

class circle:

   def __init__(self,image):
      

      self.random_pixel = (int(random.randrange(0,image.shape[1] - 50)),int(random.randrange(0,image.shape[0] - 50)))

      self.radius = 8

      # self.circled_image = cv.circle(image,self.random_pixel,8,(0,0,255),3)

      self.check = False

   def check_for_point(self,pointer_position):


      x = pointer_position[0]
      y = pointer_position[1]

      print(x), print(y)

      if (y <= (self.random_pixel[1] + 8) and y >= (self.random_pixel[1] - 8)) and (x <= (self.random_pixel[0] + 8) and x >= (self.random_pixel[0] - 8)):

         self.check = True
   
   def eaten(self,check,image):

      if self.check:

         self.random_pixel = (int(random.randrange(0,image.shape[1] - 50)),int(random.randrange(0,image.shape[0] - 50)))

         # self.circled_image = cv.circle(image,self.random_pixel,8,(0,0,255),2)

         # image.flags.writeable = True

         self.check = False
         # self.count += 1
         
circled_frame = None

with Hand_landmarker.create_from_options(options) as landmarker:

  s = 0
  if len(sys.argv) > 1:

    s = sys.argv[1]

  live_Cam = cv.VideoCapture(s)


  while cv.waitKey(1) != 27:
    
      ret,frame = live_Cam.read()
    
    #   print(frame.shape)
        
      image = frame

      frame = cv.cvtColor(frame,cv.COLOR_BGR2RGB)

      mp_format_image = mp.Image(mp.ImageFormat.SRGB,frame)

      frame_timestamp = int(cv.getTickCount() / 10000)

    #   results = "Nothing Detected"

      landmarker.detect_async(mp_format_image,frame_timestamp)
    
      modified_frame = image
      
      copy_frame = modified_frame.copy()

      if type(results) != str:
         
         pointer_point = (results[8].x * 0.64 * 1000,results[8].y * 0.48 * 1000)

         if circled_frame != None:
            
            print(circled_frame.random_pixel)

         if circled_frame == None: 

            circled_frame = circle(modified_frame)
            cv.circle(copy_frame,circled_frame.random_pixel,8,(0,0,255),2)

         circled_frame.eaten(circled_frame.check_for_point(pointer_point),modified_frame)

         # modified_frame = circled_frame.circled_image

         # print(modified_frame.flags)

         # modified_frame = cv.circle(modified_frame,circled_frame.random_pixel,8,(0,0,255),2)


      if circled_frame != None:

         cv.circle(copy_frame,circled_frame.random_pixel,8,(0,0,255),2)

      # cv.circle(copy_frame,(44,500),25,(0,0,255),-1)

      copy_frame = copy_frame[::,::-1] 

      cv.imshow("webcam",copy_frame)
      
      

