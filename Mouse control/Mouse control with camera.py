import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2 as cv
import sys
import mouse

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

    else:

        results = "Nothing Detected"

    image = output_image.numpy_view()



    

options = Hand_landmarker_options(
    base_options = base__options(model_asset_path=r"D:\Neural networks\hand_landmarker.task"),
    running_mode = Vision_running_mode.LIVE_STREAM,
    result_callback = call_backfun
)

with Hand_landmarker.create_from_options(options) as landmarker:

    s = 0
    
    if len(sys.argv) > 1:

        s = sys.argv[1]

    camera = cv.VideoCapture(s)


    while cv.waitKey(1) != 27:

        ret,frame = camera.read()

        frame = cv.cvtColor(frame,cv.COLOR_BGR2RGB)

        frame = frame[::,::-1]

        frame = cv.resize(frame,(1535,863))

        cpy_frame = frame
        
        mp_format_image = mp.Image(mp.ImageFormat.SRGB,frame)

        frame_timestamp = int(cv.getTickCount() / 10000)

        landmarker.detect_async(mp_format_image,frame_timestamp)



        if type(results) != str:

            pointer_x = int(results[8].x * frame.shape[1])
            pointer_y = int(results[8].y * frame.shape[0])

            # cv.circle(cpy_frame,(pointer_x,pointer_y),3,[255,0,0],4)

            mouse.move(pointer_x,pointer_y)
            # print([pointer_x,pointer_y])

        
        print(mouse.get_position(),end="\n\n")
        


        cpy_frame = cv.cvtColor(cpy_frame,cv.COLOR_RGB2BGR)
        # cv.imshow("Camera",cpy_frame)
            