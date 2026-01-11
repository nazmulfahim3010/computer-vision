
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
                cv2.circle(self.frame,cir,5,(255,244,0),-1)
        else:
            for cir in points:
                cv2.circle(self.frame,cir,5,(255,0,255),-1)

    def drag_and_drop(self,box_center,touch_P_A,touch_P_B,last_mid,velocity):
        circel = [touch_P_A,touch_P_B]
        dist = ((touch_P_A[0] - touch_P_B[0])**2 + (touch_P_A[1] - touch_P_B[1])**2)**0.5
        # Midpoint between fingers
        mid_x, mid_y = (touch_P_A[0] + touch_P_B[0]) // 2, (touch_P_A[1] + touch_P_B[1]) // 2

        if dist < 40: 
            if not self.is_dragging:
                # Start drag only if hand is inside the box
                if abs(mid_x - box_center[0]) < self.box_size and abs(mid_y - box_center[1]) < self.box_size:
                    self.is_dragging = True
            
            if self.is_dragging:
                velocity[0]=mid_x-last_mid[0]
                velocity[1]=mid_y-last_mid[1]
                # UPDATE BOTH X AND Y (This allows "anywhere" movement)
                box_center[0],box_center[1] = mid_x,mid_y

        else:
            self.is_dragging = False

        last_mid=[mid_x,mid_y]

        for cir in circel:
            cv2.circle(self.frame,cir,5,(145,23,75),3)

        return self.is_dragging,box_center,last_mid,velocity
    


    def distance(self,p_A,p_B):
        if not p_A or p_B:
            return
        dist = ((p_A[0]-p_B[0])**2 + (p_A[1]-p_B[1])**2)**0.5
        return dist

        ...
    def shut_down(self, p_A, p_B):
        if p_A is None or p_B is None:
            return False
        dist = ((p_A[0]-p_B[0])**2 + (p_A[1]-p_B[1])**2)**0.5
        if dist < 15:
            return True
        return False
    
    def apply_physics(self,box_center,velocity,friction,h,w):
        box_center[0] += int(velocity[0])
        box_center[1] += int(velocity[1])

        velocity[0] *= friction
        velocity[1] *= friction
#horizontal bounc logic
        # if box_center[0] - self.box_size < 0: # Hit Left
        #     box_center[0] = self.box_size
        #     velocity[0] *= -0.7 # Bounce back

        # elif box_center[0] + self.box_size > w: # Hit Right
        #     box_center[0] = w - self.box_size
        #     velocity[0] *= -0.7

        # # Top and Bottom Walls
        # if box_center[1] - self.box_size < 0: # Hit Top
        #     box_center[1] = self.box_size
        #     velocity[1] *= -0.7

        # elif box_center[1] + self.box_size > h: # Hit Bottom
        #     box_center[1] = h - self.box_size
        #     velocity[1] *= -0.7
        return box_center,velocity
        ...
