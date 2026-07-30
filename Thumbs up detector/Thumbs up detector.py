import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2 as cv
import sys

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

def thumbs_up_check(handlandmarks):

    thumb_y = handlandmarks[3].y

    check = True

    window_open_check = cv.getWindowProperty("thumbs up",cv.WND_PROP_VISIBLE)


    # print(thumb_y)

    for i in handlandmarks:

       if i.y == thumb_y or i.y == handlandmarks[4].y:
          continue

       if i.y < thumb_y:
          check = False
          break
    
    if not (handlandmarks[8].x < handlandmarks[4].x + 0.1) or not (handlandmarks[8].x > handlandmarks[4].x - 0.1):
       
       check = False
    if not (handlandmarks[12].x < handlandmarks[4].x + 0.1) or not (handlandmarks[12].x > handlandmarks[4].x - 0.1):
      
       check = False
    if not (handlandmarks[16].x < handlandmarks[4].x + 0.1) or not (handlandmarks[16].x > handlandmarks[4].x - 0.1):
      
       check = False
   #  print("\n\n")

    if check and window_open_check != -1:
       
       cv.imshow("thumbs up",thumbs_up)
    
    elif check == False and window_open_check == 1:
       
       cv.destroyWindow("thumbs up")

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
      
      if type(results) != str:
         
         for i in range(len(results)):

            if i != len(results) - 1:
               
               pt1 = (int(results[i].x * 1000 * 0.64),int(results[i].y * 1000 * 0.48))
               pt2 = (int(results[i+1].x * 1000 * 0.64),int(results[i+1].y * 1000 * 0.48))

              #  print(pt1)
              #  print(pt2)
            
            modified_frame.flags.writeable= True
            modified_frame = cv.line(modified_frame,pt1,pt2,(0,0,255),2)

         thumbs_up_check(results)


      cv.imshow("webcam",modified_frame)
      
      

