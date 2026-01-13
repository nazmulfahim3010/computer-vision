import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pack import track_custom_nazz

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
        
        box_center=[300,300]
        box_size=30
        is_dragging =False
        timestamp_ms = 0

        # physics

        last_mid =[300,300]
        velocity=[0,0]
        friction = 0.95

        while cam.isOpened:
            success , frame = cam.read()
            if success == False:
                break

            h, w,_ = frame.shape
            frame = cv2.flip(frame,1) 
            frameRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = frameRGB)
            # timestamp_ms = int(cam.get(cv2.CAP_PROP_POS_MSEC))
            timestamp_ms+=33
            result = landmarker.detect_for_video(mp_image,timestamp_ms)
            
            HAND_CONNECTIONS = [
                (0, 1), (1, 2), (2, 3), (3, 4),    # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8),    # Index Finger
                (5, 9), (9, 10), (10, 11), (11, 12), # Middle Finger
                (9, 13), (13, 14), (14, 15), (15, 16), # Ring Finger
                (13, 17), (17, 18), (18, 19), (19, 20), (0, 17) # Pinky and Palm
            ]

            box_x1,box_y1 = w-250 , 50
            box_x2,box_y2 = w-50 ,250

            l_p0,r_p8 = None,None
            if result.hand_landmarks:
                for hand_id, (hand_landmarks, hand) in enumerate(zip(result.hand_landmarks, result.handedness)):
                    

                    hand_label = hand[0].category_name
                    hand_label = 'Left' if hand_label == 'Right' else 'Right'
                    pipe = track_custom_nazz(frame,hand_label,box_size,is_dragging)
                    
                    points = []
                    for lm in hand_landmarks:
                        points.append((int(lm.x * w), int(lm.y * h)))

                    if hand_label == 'Left': l_p0 = points[0]
                    if hand_label == 'Right': r_p8 = points[8]
                        
                    pipe.art_hand(points)
                    pipe.paint_dot(points)

                 
                    if hand_label == 'Left': 
                        p4, p8 = points[12],points[8]
                        is_dragging,box_center,last_mid,velocity = pipe.drag_and_drop(
                            box_center,
                            p4,p8,
                            last_mid,
                            velocity)
                
            if not is_dragging:
                phy_pipe=track_custom_nazz(frame,box_size)

                box_center,velocity = phy_pipe.apply_physics(box_center,velocity,friction,h,w)


            #box drwing
            top_left = (box_center[0] - box_size, box_center[1] - box_size)
            bottom_right = (box_center[0] + box_size, box_center[1] + box_size)

            
            box_color = (255, 0, 255) if is_dragging else (255, 255, 255)
            cv2.rectangle(frame, top_left, bottom_right, box_color, cv2.FILLED if is_dragging else 3)  

            
               
            if l_p0 and r_p8:
                checker = track_custom_nazz(frame)
                if checker.shut_down(l_p0, r_p8):
                    print("Shutdown gesture detected!")
                    break 
            hit_border = False
            
            # --- Left and Right Walls ---
            if box_center[0] - box_size <= 0: # Hit Left
                box_center[0] = box_size
                velocity[0] *= -0.7 
                hit_border = True
            elif box_center[0] + box_size >= w: # Hit Right
                box_center[0] = w - box_size  # Fixed: Snap to right
                velocity[0] *= -0.7 
                hit_border = True

            # --- Top and Bottom Walls ---
            if box_center[1] - box_size <= 0: # Hit Top
                box_center[1] = box_size
                velocity[1] *= -0.7
                hit_border = True
            elif box_center[1] + box_size >= h: # Hit Bottom
                box_center[1] = h - box_size  # Fixed: Snap to bottom
                velocity[1] *= -0.7
                hit_border = True

            # Draw a border at the very edge of the camera feed
            border_color = (0, 0, 255) if hit_border else (0, 255, 255)
            border_thickness = 10 if hit_border else 2
            cv2.rectangle(frame, (0, 0), (w, h), border_color, border_thickness)

            cv2.imshow('window',frame)
            # cv2.imshow('RGB',frameRGB)
            # print(result.hand_landmarks)
            
            if cv2.waitKey(1) & 0xFF == 27: #esc
                break

        cam.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
