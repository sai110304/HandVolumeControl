import cv2
import mediapipe as mp
import numpy as np
import time
import alsaaudio


#######################################################################################################################
def get_volume_info():
    mixer = alsaaudio.Mixer() 
    volume = mixer.getvolume()[0] 
    mute_status = mixer.getmute()[0]
    return volume, mute_status

def set_volume(volume_percentage):
    mixer = alsaaudio.Mixer() 
    mixer.setvolume(volume_percentage)  # Set volume for all channels


def toggle_mute():
    mixer = alsaaudio.Mixer() 
    mute_status = mixer.getmute()[0]  # Get the current mute status
    mixer.setmute(not mute_status)

#######################################################################################################################

camera_width = 640
camera_height = 720

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands=mp_hands.Hands(min_detection_confidence=0.75, max_num_hands=1)


def process_image(img):
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # converting BGR to RGB since mp_hands takes only RGB images
    outputs = hands.process(RGB_img)
    return outputs



def find_hands(img, outputs):
    if outputs.multi_hand_landmarks:
        for single_hand_landmarks in outputs.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, single_hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return img



def find_handlandmark_position(img, outputs, hand_no=0):
    landmark_list = []
    if outputs.multi_hand_landmarks:
        my_hand = outputs.multi_hand_landmarks[hand_no]
        for id, land_mark in enumerate(my_hand.landmark):
            h, w, c = img.shape
            cx, cy = int(land_mark.x * w), int(land_mark.y * h)   #converting normalized values to pixel values
            landmark_list.append([id, cx, cy])
            
   
    if landmark_list:
        landmark_array = np.array(landmark_list)
        min_x, min_y = np.min(landmark_array[:, 1]), np.min(landmark_array[:, 2])
        max_x, max_y = np.max(landmark_array[:, 1]), np.max(landmark_array[:, 2])
        bounding_box = [min_x, min_y, max_x, max_y]
    else:
        bounding_box = [None, None, None, None]
        
    return landmark_list, bounding_box


#######################################################################################################################

pt,ct=0,0  #previous time,current time

url = 'http://192.0.0.4:8080/video'  #url of the ip webcam app
cap = cv2.VideoCapture(url)

cap.set(3, camera_width)
cap.set(4, camera_height)

volBar = 400
vol=0
draw=True
muted_info=False
while True:
    success, img = cap.read()
    outputs=process_image(img)
    img=find_hands(img,outputs)
    LandMarkList,bounding=find_handlandmark_position(img,outputs)
    if draw==True:
        if bounding[0]!=None:
            cv2.rectangle(img, (bounding[0]-20, bounding[1]-20), (bounding[2]+20, bounding[3]+20), (0, 255, 0), 2)
        
    if len(LandMarkList)!=0:
        
        bb_height=bounding[3]-bounding[1]
        bb_width=bounding[2]-bounding[0]
        bb_area=(bb_height*bb_width)//100
        print(bb_area)
        if bb_area>600 and bb_area <2500:
            # print(LandMarkList[4],LandMarkList[8])   #4th and 8th landmark are thumb tip and index finger tip respectively 
            cv2.circle(img, (LandMarkList[4][1], LandMarkList[4][2]), 15, (255, 255, 255), cv2.FILLED)
            cv2.circle(img, (LandMarkList[8][1], LandMarkList[8][2]), 15, (255, 255, 255), cv2.FILLED)
            cv2.line(img, (LandMarkList[4][1], LandMarkList[4][2]), (LandMarkList[8][1], LandMarkList[8][2]), (255, 255, 0), 3)
            length=np.hypot(LandMarkList[4][1]-LandMarkList[8][1],LandMarkList[4][2]-LandMarkList[8][2]) #distance between thumb tip and index finger tip

            vol=np.interp(length,[45,500],[0,100])
            vol=2 * round(vol/2)
            x,y =LandMarkList[20][1],LandMarkList[20][2]  #finger tip of small finger
            cv2.circle(img,(x,y),15, (255, 0, 255), cv2.FILLED)
            if (LandMarkList[20][2] > LandMarkList[18][2]):
                if vol>0 and muted_info==True: toggle_mute()
                set_volume(volume_percentage=int(vol))
                # print(vol,length)
        
        
    volume,muted_info=get_volume_info()
    if muted_info==True: volume=0
    volBar = np.interp(vol, [0, 100], [400, 150])
    cv2.rectangle(img, (50, 150), (85, 400), (255, 255, 255), 2)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 255, 255), cv2.FILLED)
    cv2.putText(img, f'Control Volume: {int(vol)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,1, (255, 255, 255), 2)
    cv2.putText(img, f' Volume Set: {int(volume)} %', (700, 50), cv2.FONT_HERSHEY_COMPLEX,1, (255, 255, 255), 2)
    ct = time.time()
    fps = 1 / (ct - pt)
    pt = ct
    cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('Output', img)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to exit
        break    