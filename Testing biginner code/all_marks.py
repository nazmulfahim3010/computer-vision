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
                (0, 1), (1, 2), (2, 3), (3, 4),(4,8),    # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8),    # Index Finger
                (5, 9), (9, 10), (10, 11), (11, 12), # Middle Finger
                (9, 13), (13, 14), (14, 15), (15, 16), # Ring Finger
                (13, 17), (17, 18), (18, 19), (19, 20), (0, 17) # Pinky and Palm
            ]

            if result.hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(result.hand_landmarks):
                    print(f"----------{hand_idx+1}-----------")

                    point=[]
                    for pos, land_mark in enumerate(hand_landmarks):
                        h, w, _ = frame.shape
                        # print(point)
                        point.append((int(land_mark.x*w),int(land_mark.y*h))) 

                        # print(f'-->{pos}  x:{x_pixel} y:{y_pixel}')
                        # cv2.circle(frame,(x_pixel,y_pixel),5,(0,0,255),-1)
                            
                    for connection in HAND_CONNECTIONS:
                            str_ind=connection[0]
                            end_ind=connection[1]

                            cv2.line(frame,point[str_ind],point[end_ind],(0,255,0),2)
                        
                    for cir in point:
                            cv2.circle(frame,cir,5,(0,0,255),-1)
            cv2.imshow('window',frame)
            # cv2.imshow('RGB',frameRGB)
            # print(result.hand_landmarks)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

       
        cam.release()

    


if __name__ == '__main__':
    main()
    
    cv2.destroyAllWindows()
