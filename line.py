import cv2
import numpy as np
import math

# capture the video from camera
cap = cv2.VideoCapture(1)


while True:
    # read the frame from the camera
    ret, frame = cap.read()
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    lower_tape = np.array([0,153,153]) # [0, 0, 0]
    upper_tape = np.array([40,255,255]) # [50, 50, 50]
    mask = cv2.inRange(frame, lower_tape, upper_tape)
    mask_yellow=cv2.inRange(hsv,lower_tape,upper_tape)

    kernel=np.ones((3,3),np.uint8)
    mask=cv2.erode(mask,kernel,iterations=1)
    mask=cv2.dilate(mask,kernel,iterations=9)
    '''if cv2.countNonZero(mask) == 0:
         cv2.putText(frame,("stop"),(10,400),cv2.FONT_HERSHEY_SIMPLEX,1,(200,100,255),2)'''
    if cv2.countNonZero(mask_yellow)==0:
        cv2.putText(frame,("stop"),(10,400),cv2.FONT_HERSHEY_SIMPLEX,1,(200,100,255),2)

    contours, hierarchy=cv2.findContours(mask_yellow,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(frame,contours,-1,(0,255,0),3) 

    if len(contours)>0:
        x,y,w,h=cv2.boundingRect(contours[0])
        #finding minimum area rotated, returning (center(x,y),(w,h),angle of rotation)
        min_box=cv2.minAreaRect(contours[0])
        
        (x_min,y_min),(w_min,h_min),angle=min_box
        box=cv2.boxPoints(min_box)
        box=np.intp(box)
        angle=int(angle)
        

        frame_center_x = frame.shape[1] // 2 
        frame_center_y = frame.shape[0] // 2
        cv2.circle(frame,(frame_center_x,frame_center_y),5,(255,0,0),-1) #mid of screen, blue
        #cv2.circle(frame,(frame_center_x-50,frame_center_y),5,(255,0,0),-1) #mid left of screen, blue
        cv2.circle(frame,(frame_center_x+50,frame_center_y),5,(255,0,0),-1) #mid right of screen, blue
        cv2.circle(frame,(int(x_min),int(y_min)),5,(0,255,0),-1) # mid of bounding box, green

# put this in a while loop to break when it goes straight 
        if x_min<(frame_center_x-50):
            cv2.putText(frame,("straft right"),(300,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
        elif x_min>(frame_center_x+50):
            cv2.putText(frame,("straft left"),(300,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
        else:
            cv2.putText(frame,("in range"),(300,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

 
        if angle<45: 
            angle=90-angle
            
        else:
            angle=180-angle
        
        cv2.drawContours(frame,[box],-1,(0,0,255),3)
        cv2.putText(frame,("angle="+str(angle)),(10,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)


        if angle>=80 and angle<=100:
            cv2.putText(frame,("go straight"),(10,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        elif angle<80:
            cv2.putText(frame,("turn right"),(10,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        elif angle>100:
            cv2.putText(frame,("turn left"),(10,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        else:
            cv2.putText(frame,("stop"),(10,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    cv2.resize(frame,(640,480))
    cv2.imshow("frame",frame)
    #cv2.resize(mask,(640,480))
    #cv2.imshow("mask",mask)

    cv2.resize(hsv,(640,480))
    cv2.imshow("mask",mask_yellow)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the camera and close the windows
cap.release()
cv2.destroyAllWindows()
