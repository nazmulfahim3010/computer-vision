import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def main():
        MODEL_PATH='models/hand_landmarker.task'

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker= vision.HandLandmarker
        HandLandmarkerOptions= vision.HandLandmarkerOptions
        VisionRunningMode= vision.RunningMode


        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,
            num_hands = 2
        )

        landmarker = HandLandmarker.create_from_options(options)

        cam = cv2.VideoCapture(0)
        
        while cam.isOpened:
            success , frame = cam.read()

            frame = cv2.flip(frame,1) 
            if success == False:
                break
            frameRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = frameRGB)
            timestamp_ms = int(cam.get(cv2.CAP_PROP_POS_MSEC))
            result = landmarker.detect_for_video(mp_image,timestamp_ms)
            
            HAND_CONNECTIONS = [
                (0, 1), (1, 2), (2, 3), (3, 4),    # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8),    # Index Finger
                (5, 9), (9, 10), (10, 11), (11, 12), # Middle Finger
                (9, 13), (13, 14), (14, 15), (15, 16), # Ring Finger
                (13, 17), (17, 18), (18, 19), (19, 20), (0, 17) # Pinky and Palm
            ]

            if result.hand_landmarks:
                 #handedness mean type of hand it is left or right
                 for hand_id,(hand_landmarks,hand) in enumerate(zip(result.hand_landmarks,result.handedness)):
                      
                    hand_label = hand[0].category_name
                    print(hand_label)

                    if hand_label == 'Left':
                        ...
                    points = []
                    for lm in hand_landmarks:
                         h, w, _ = frame.shape
                         points.append((int(lm.x*w),int(lm.y*h)))
                    
                    for connection in HAND_CONNECTIONS:
                         A=points[connection[0]]
                         B=points[connection[1]]

                         cv2.line(frame,A,B,(0,255,0),2)

                    if hand_label =='Left':
                        tag=points[19]
                        cv2.putText(frame,f"This is hand text",(tag[0],tag[1]-20),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                    else:
                         tag = points[4]
                         cv2.putText(frame,f"This is Right Hand",(tag[0],tag[1]-20),
                                     cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)

                      


            cv2.imshow('window',frame)
            # cv2.imshow('RGB',frameRGB)
            # print(result.hand_landmarks)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

       
        cam.release()

    
def left_hand_functions():
     pass

if __name__ == '__main__':
    main()
    
    cv2.destroyAllWindows()
