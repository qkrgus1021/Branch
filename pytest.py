import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import pygame
mp_holistic = mp.solutions.holistic #holistic model
mp_drawing = mp.solutions.drawing_utils #drawing utilites
def mediapipe_detection(image,model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #BGR 2 RGB
    image.flags.writeable = False                 #image is no longer writeable
    results = model.process(image)                 #make prediction
    image.flags.writeable = True                  #image is writeable
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR) #RGB 2 BGR
    return image,results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image,results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) #draw pose connections
    
def extract_keypoints(results):
    pose =[]
    count=0
    for res in results.pose_landmarks.landmark:
        if(count==0):
            test = np.array([res.x, res.y,res.z])
            pose.append(test)
        count+=1
    pose = np.array(pose).flatten()
    return pose    

def extract_badkeypoints(results):
    pose =[]
    pose2 =[]
    pose3 =[]
    pose4 =[]
    pose5 =[]
    pose6 =[]
    pose7 =[]
    pose8 =[]
    #대각선 4사분면
    pose=(results[0]+0.15,results[1]+0.15,results[2])
    #대각선 2사분면
    pose2=(results[0]-0.15,results[1]-0.15,results[2])
    #대각선 1사분면
    pose3=(results[0]+0.15,results[1]-0.15,results[2])
    #대각성 3사분면
    pose4=(results[0]-0.15,results[1]+0.15,results[2])
    #x축
    pose5=(results[0]+0.12,results[1],results[2])
    pose6=(results[0]-0.12,results[1],results[2])
    #y축
    pose7=(results[0],results[1]+0.05,results[2])
    pose8=(results[0],results[1]-0.05,results[2])
    
    pose = np.array(pose).flatten()
    pose2 = np.array(pose2).flatten()
    pose3 = np.array(pose3).flatten()
    pose4 = np.array(pose4).flatten()
    pose5 = np.array(pose5).flatten()
    pose6 = np.array(pose6).flatten()
    pose7 = np.array(pose7).flatten()
    pose8 = np.array(pose8).flatten()
    return pose, pose2, pose3, pose4,pose5,pose6,pose7,pose8

colors = [(245,117,16),(117,245,16)]

def prob_viz(res,actions, input_frame, colors):
    output_frame = input_frame.copy()
    if np.argmax(res)==0:
        num=0
        prob=res[np.argmax(res)]
    else:
        num=1
        prob=res[np.argmax(res)]   
    cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100),90+num*40),colors[num],-1)
    cv2.putText(output_frame,actions[num],(0,85+num*40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
    return output_frame

#touch event
def showBlank(event, x, y, flags, param):
    #param is the array i from below
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.imshow('OpenCV Feed',image)
        param[0] = param[0] + 1
    elif event == cv2.EVENT_RBUTTONDOWN:
        cv2.imshow('OpenCV Feed',image)
        param[0] = param[0] + 1

# path for exproted data
DATA_PATH = os.path.join("MP_DATA")

#Action that we try to detect
actions = np.array(['good','bad','bad2','bad3','bad4','bad5','bad6','bad7','bad8'])

#thirty videos worth of data
no_sequences = 5

#videos are goint to be 30 frames in length
sequence_length =10

for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass
        
cap = cv2.VideoCapture(0)
cv2.namedWindow("OpenCV Feed", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("OpenCV Feed",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

#set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    #LOOP through sequneces aka videos
    for sequence in range(no_sequences):
        #Loop through video length aka sequnece length
        for frame_num in range(sequence_length):
                
            #Read feed
            ret,frame = cap.read()
            
            
            #make detection
            image , results  = mediapipe_detection(frame,holistic)
            draw_landmarks(image,results)
            
            #프로그램을 시작할 때 메시지와 음성을 출력해준다.
            if sequence == 0 and frame_num==0:
                i = [0]
                cv2.putText(image,"Touch the Screen.",(100,100),cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,0,255), 4, cv2.LINE_AA)
                cv2.imshow("OpenCV Feed",image)
                
                pygame.mixer.init()
                pygame.mixer.music.load('start.mp3')
                pygame.mixer.music.play()
                # the mousecallback only needs to be set once
                cv2.setMouseCallback("OpenCV Feed", showBlank, i )
                # show the initial image for the first time.
                while i[0] < 1:    
                    cv2.waitKey(100)
                    cv2.putText(image,"Touch the Screen.",(100,100),cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,0,255), 4, cv2.LINE_AA)
                    cv2.imshow("OpenCV Feed",image)
                    
                pygame.mixer.init()
                pygame.mixer.music.load('fixing.mp3')
                pygame.mixer.music.play()
            #사용자의 모습이 보이지 않았을 경우 r키를 입력하여 다시 실행할 수 있게 한다.
            if type(results.pose_landmarks) == type(None):
                frame_num=frame_num-1
                i=[0]
                cv2.setMouseCallback("OpenCV Feed", showBlank, i )
                while i[0] < 1:    
                    cv2.waitKey(100)
                    cv2.putText(image,"Touch the Screen.",(100,100),cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,0,255), 4, cv2.LINE_AA)
                    cv2.imshow("OpenCV Feed",image)
                continue
            
            #Apply collection logic
            if frame_num == 0:
                cv2.putText(image, 'STARTING COLLECTION', (120,200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 4, cv2.LINE_AA)
                    
                cv2.putText(image, 'Collectiong frames for {} Video Number {}'.format(action,sequence), (15,12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 4, cv2.LINE_AA)
                #show to screen
                cv2.imshow("OpenCV Feed",image)
                cv2.waitKey(1000)
            else:
                cv2.putText(image, 'Collectiong frames for {} Video Number {}'.format(action,sequence), (15,12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 4,cv2.LINE_AA)
                    
            #show to screen
            cv2.imshow("OpenCV Feed",image)
                  
            #new export keypoints
            keypoints = extract_keypoints(results)
            npy_path=os.path.join(DATA_PATH, 'good', str(sequence),str(frame_num))
            np.save(npy_path,keypoints)
            badkeypoints1,badkeypoints2, badkeypoints3, badkeypoints4, badkeypoints5, badkeypoints6, badkeypoints7, badkeypoints8 = extract_badkeypoints(keypoints)
            npy_path=os.path.join(DATA_PATH, 'bad', str(sequence),str(frame_num))
            np.save(npy_path,badkeypoints1)
            npy_path=os.path.join(DATA_PATH, 'bad2', str(sequence),str(frame_num))
            np.save(npy_path,badkeypoints2)
            npy_path=os.path.join(DATA_PATH, 'bad3', str(sequence),str(frame_num))
            np.save(npy_path,badkeypoints3)
            npy_path=os.path.join(DATA_PATH, 'bad4', str(sequence),str(frame_num))
            np.save(npy_path,badkeypoints4)
            npy_path=os.path.join(DATA_PATH, 'bad5', str(sequence),str(frame_num))
            np.save(npy_path,badkeypoints5)
            npy_path=os.path.join(DATA_PATH, 'bad6', str(sequence),str(frame_num))
            np.save(npy_path,badkeypoints6)
            npy_path=os.path.join(DATA_PATH, 'bad7', str(sequence),str(frame_num))
            np.save(npy_path,badkeypoints7)
            npy_path=os.path.join(DATA_PATH, 'bad8', str(sequence),str(frame_num))
            np.save(npy_path,badkeypoints8)
    
cap.release()
cv2.destroyAllWindows()

#build train LSTM
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

label_map = {label:num for num, label in enumerate(actions)}

sequences , labels = [],[]
for action in actions:
    for sequence in range(no_sequences):
        window=[]
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH,action,str(sequence),"{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y= to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64,return_sequences=True,activation='relu',input_shape=(10,3)))
model.add(LSTM(128,return_sequences=True,activation='relu'))
model.add(LSTM(64,return_sequences=False,activation='relu'))

model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['categorical_accuracy'])
#crossentropy -> 수치로 표시하기에 유리한 방식으로 출력해주기 때문이다.
#왜 이러한 구조로 구성하였나? -> 
#1. 적은 양의 데이터만 사용할 예정이고
#2. 빠르게 학습시킬 수 있다는 장점과
#3. 실시간으로 평가를 빠르게 내려줄 수 있기 때문입니다.
model.fit(X_train,y_train,epochs=150,callbacks=[tb_callback])

model.save('action.h5')




#1 Net detection variables

sequence = []
sentence = []
predictions=[]
threshold = 0.4
bad_pose_count=0
start =0

cap = cv2.VideoCapture(0)
cv2.namedWindow("OpenCV Feed", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("OpenCV Feed",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

#set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        #Read feed
        ret,frame = cap.read()
        
        if not ret:
            continue
            
        #make detection
        image , results  = mediapipe_detection(frame,holistic)
        #draw_landmark
        #사용자의 모습이 보이지 않았을 경우 r키를 입력하여 다시 실행할 수 있게 한다.
        if type(results.pose_landmarks) == type(None):
            i = [0]
            cv2.putText(image,"Touch the Screen.",(100,100),cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,0,255), 4, cv2.LINE_AA)
            cv2.imshow("OpenCV Feed",image)
            cv2.setMouseCallback('OpenCV Feed', showBlank, i )
            while i[0] < 1:    
                cv2.waitKey(100)
                cv2.putText(image,"Touch the Screen.",(100,100),cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,0,255), 4, cv2.LINE_AA)
                cv2.imshow("OpenCV Feed",image)
            continue
                
        keypoints = extract_keypoints(results)
        sequence.insert(0,keypoints)
        sequence = sequence[:10]
        
        if len(sequence) == 10:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
        
            #3. vizs logic - 0.4보다 큰 수치를 가졌을 경우에 상태가 바뀌면 바뀐 상태로, 안바뀌면 안바뀐 상태로 
            if res[np.argmax(res)] > threshold:
                if len(sentence) > 0:
                    if actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                    #나쁜 자세가 5분 정도 지속적으로 유지되었을 때 음성으로 알려준다.
                    if actions[np.argmax(res)] !="good" and start == 0:
                        start = time.time()
                    elif actions[np.argmax(res)] == "good":
                        start = 0
                    elif actions[np.argmax(res)] !="good" :
                        dif=time.time()-start
                        if dif > 60:
                            pygame.mixer.init()
                            pygame.mixer.music.load('good.mp3')
                            pygame.mixer.music.play()
                            start=time.time()

                else:
                    sentence.append(actions[np.argmax(res)])

            if len(sentence)>5:
                sentence = sentence[-5:]
            
            #viz
            image = prob_viz(res,actions,image,colors)
            #show to screen
            cv2.imshow("OpenCV Feed",image)
            
        #breaking
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
cap.release()
cv2.destroyAllWindows()




