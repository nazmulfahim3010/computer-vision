import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from __init__ import track_custom_nazz

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
        box_size=50
        is_dragging =False

        while cam.isOpened:
            success , frame = cam.read()
            h, w,_ = frame.shape

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

            box_x1,box_y1 = w-250 , 50
            box_x2,box_y2 = w-50 ,250

            
            if result.hand_landmarks:
                for hand_id, (hand_landmarks, hand) in enumerate(zip(result.hand_landmarks, result.handedness)):
                    

                    hand_label = hand[0].category_name
                    hand_label = 'Left' if hand_label == 'Right' else 'Right'
                    pipe = track_custom_nazz(frame,hand_label,box_size,is_dragging)
                    
                    points = []
                    for lm in hand_landmarks:
                        points.append((int(lm.x * w), int(lm.y * h)))
                    
                    pipe.art_hand(points,None)
                    pipe.paint_dot(points)

                    
                    if hand_label == 'Right':
                        
                        p4, p8 = points[4],points[8]
                        is_dragging,box_center = pipe.drag_and_drop(box_center,p4,p8)
                        
            top_left = (box_center[0] - box_size, box_center[1] - box_size)
            bottom_right = (box_center[0] + box_size, box_center[1] + box_size)
            
            box_color = (255, 0, 255) if is_dragging else (255, 255, 255)
            cv2.rectangle(frame, top_left, bottom_right, box_color, cv2.FILLED if is_dragging else 3)                    

            cv2.imshow('window',frame)
            # cv2.imshow('RGB',frameRGB)
            # print(result.hand_landmarks)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cam.release()
        cv2.destroyAllWindows()


# def art_hand(HAND_CONNECTIONS,frame,hand_label,points):
#     for connection in HAND_CONNECTIONS:
#         A=points[connection[0]]
#         B=points[connection[1]]

#         cv2.line(frame,A,B,(0,255,0),2)

#         # if hand_label =='Left':
#         #     tag=points[19]
#         #     cv2.putText(frame,f"This is Right Hand",(tag[0],tag[1]-20),
#         #                 cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
#         # else:
#         #         tag = points[4]
#         #         cv2.putText(frame,f"This is Left Hand",(tag[0],tag[1]-20),
#         #                     cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2) 

# def paint_dot(points,hand_label,frame):
#     if hand_label == 'Left':
#         for cir in points:
#             cv2.circle(frame,cir,5,(255,0,0),-1)
    # else:
    #     for cir in points:
    #         cv2.circle(frame,cir,5,(255,0,255),-1)

# def drag_and_drop(frame,is_dragging,box_center,box_size,p4,p8):
#     dist = ((p4[0] - p8[0])**2 + (p4[1] - p8[1])**2)**0.5
#     # Midpoint between fingers
#     mid_x, mid_y = (p4[0] + p8[0]) // 2, (p4[1] + p8[1]) // 2

#     if dist < 40: 
#         if not is_dragging:
#             # Start drag only if hand is inside the box
#             if abs(mid_x - box_center[0]) < box_size and abs(mid_y - box_center[1]) < box_size:
#                 is_dragging = True
        
#         if is_dragging:
#             # UPDATE BOTH X AND Y (This allows "anywhere" movement)
#             box_center[0] = mid_x
#             box_center[1] = mid_y
#     else:
#         is_dragging = False

#     return is_dragging,box_center

    # 4. Draw the Box (Outside hand loop for stability)
    # Ensure the corners use both [0] and [1]
   




if __name__ == '__main__':
    main()
