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
            
        
            if result.hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(result.hand_landmarks):
                    print(f"----------{hand_idx+1}-----------")

                    line_data={}

                    for point, land_mark in enumerate(hand_landmarks):
                        h, w, _ = frame.shape
                        # print(point)
                        if point==8 or point == 4 or point == 0:
                            print(point)
                            x_pixel,y_pixel, z_value = int(land_mark.x*w),int(land_mark.y*h),int(land_mark.z)

                            line_data[point]=(x_pixel,y_pixel)

                            print(f'-->{point}  x:{x_pixel} y:{y_pixel}')
                            cv2.circle(frame,(x_pixel,y_pixel),5,(0,0,255),-1)
                            

                    if len(line_data):
                        p8=line_data[8]
                        p4=line_data[4]
                    

                        distance=((p8[0]-p4[0])**2+(p8[1]-p4[1])**2)**0.5
                        cv2.line(frame,p4,p8,(0,155,0),2)
                        print(int(distance))
                        cv2.putText(frame, f"Dist: {int(distance)}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        if distance <= 30 and distance>=10:
                            cam.release()
                            cv2.destroyAllWindows()
                        # if distance < 30:
                        #     cv2.putText(frame, "Halland!", (200, 200), 
                        #     cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

            cv2.imshow('window',frame)
            # cv2.imshow('RGB',frameRGB)
            # print(result.hand_landmarks)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cam.release()
        cv2.destroyAllWindows()


    


if __name__ == '__main__':
    main()
