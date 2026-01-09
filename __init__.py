
import cv2
import math

class track_custom_nazz:
    def __init__(self,frame,hand_label=None,box_size=None,is_dragging=False):
        self.frame = frame
        self.HAND_CONNECTIONS=HAND_CONNECTIONS = [
                (0, 1), (1, 2), (2, 3), (3, 4),    # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8),    # Index Finger
                (5, 9), (9, 10), (10, 11), (11, 12), # Middle Finger
                (9, 13), (13, 14), (14, 15), (15, 16), # Ring Finger
                (13, 17), (17, 18), (18, 19), (19, 20), (0, 17) # Pinky and Palm
            ]
        self.hand_label=hand_label
        self.is_dragging=is_dragging
        self.box_size = box_size

    
    def art_hand(self,points,R_text=None,L_text=None):
        for connection in self.HAND_CONNECTIONS:
            A=points[connection[0]]
            B=points[connection[1]]

            cv2.line(self.frame,A,B,(0,255,0),2)

            if R_text or L_text:
                if self.hand_label =='Left':
                    tag=points[19]
                    cv2.putText(self.frame,R_text,(tag[0],tag[1]-20),
                                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                else:
                        tag = points[4]
                        cv2.putText(self.frame,L_text,(tag[0],tag[1]-20),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2) 

    def paint_dot(self,points,ALL=None):
        if self.hand_label == 'Left':
            for cir in points:
                cv2.circle(self.frame,cir,5,(255,0,0),-1)
        else:
            for cir in points:
                cv2.circle(self.frame,cir,5,(255,0,255),-1)

    def drag_and_drop(self,box_center,touch_p_A,touch_P_B):
        
        dist = ((touch_p_A[0] - touch_P_B[0])**2 + (touch_p_A[1] - touch_P_B[1])**2)**0.5
        # Midpoint between fingers
        mid_x, mid_y = (touch_p_A[0] + touch_P_B[0]) // 2, (touch_p_A[1] + touch_P_B[1]) // 2

        if dist < 40: 
            if not self.is_dragging:
                # Start drag only if hand is inside the box
                if abs(mid_x - box_center[0]) < self.box_size and abs(mid_y - box_center[1]) < self.box_size:
                    self.is_dragging = True
            
            if self.is_dragging:
                # UPDATE BOTH X AND Y (This allows "anywhere" movement)
                box_center[0] = mid_x
                box_center[1] = mid_y
        else:
            self.is_dragging = False

        return self.is_dragging,box_center
        ...
    
         