import pygame
import sys
from pygame.locals import QUIT

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import pygame
import datetime
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

def Hextract_badkeypoints(results,x,y):
    pose =[]
    pose2 =[]
    pose3 =[]
    pose4 =[]
    pose5 =[]
    pose6 =[]
    pose7 =[]
    pose8 =[]
    #대각선 4사분면
    pose=(results[0]+x,results[1]+x,results[2])
    #대각선 2사분면
    pose2=(results[0]-x,results[1]-x,results[2])
    #대각선 1사분면
    pose3=(results[0]+x,results[1]-x,results[2])
    #대각성 3사분면
    pose4=(results[0]-x,results[1]+x,results[2])
    #x축
    pose5=(results[0]+x,results[1],results[2])
    pose6=(results[0]-x,results[1],results[2])
    #y축
    pose7=(results[0],results[1]+y,results[2])
    pose8=(results[0],results[1]-y,results[2])
    
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

mode='mode0'


def prob_viz(res,actions, input_frame, colors,mode,counter,now):
    output_frame = input_frame.copy()
    if np.argmax(res)==0:
        num=0
        prob=res[np.argmax(res)]
    else:
        num=1
        prob=res[np.argmax(res)]   
    cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100),90+num*40),colors[num],-1)
    cv2.putText(output_frame,actions[num],(0,85+num*40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(output_frame,mode,(0,85+2*40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)
    cv2.putText(output_frame,str(counter),(0,85+3*40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)
    cv2.putText(output_frame,"{} : {} : {}".format(now.hour,now.minute,now.second),(0,460),cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,0), 4, cv2.LINE_AA)
    return output_frame

#touch event
def showBlank(event, x, y, flags, param):
    #param is the array i from below
    if event == cv2.EVENT_LBUTTONDOWN:
        param[0] = param[0] + 1
    elif event == cv2.EVENT_RBUTTONDOWN:
        param[0] = param[0] + 1
    if x>120 and x<188 and y<25:
        param[0]=-10
    elif x>239 and x<324 and y<25:
        param[0]=-20
    elif x>375 and x<433 and y<25:
        param[0]=-30
    elif x>511 and x<562 and y<25:
        param[0]=-40
        
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

def train(x,y):
    i = [0]
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("OpenCV Feed", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("OpenCV Feed",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback("OpenCV Feed", showBlank, i )
    pygame.mixer.init()
    #set mediapipe model
    with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:
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
                    cv2.putText(image,"Touch the Screen.",(100,100),cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,0,255), 4, cv2.LINE_AA)
                    cv2.imshow("OpenCV Feed",image)
                    pygame.mixer.music.load('start.mp3')
                    pygame.mixer.music.play()
                    # show the initial image for the first time.
                    while i[0] < 1:    
                        cv2.waitKey(10)
                        cv2.imshow("OpenCV Feed",image)


                #사용자의 모습이 보이지 않았을 경우 r키를 입력하여 다시 실행할 수 있게 한다.
                try:
                    keypoints = extract_keypoints(results)
                    i[0]=5
                except Exception as e:
                    i[0]=0
                    cv2.putText(image,"Touch the Screen.",(100,100),cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,0,255), 4, cv2.LINE_AA)
                    cv2.imshow("OpenCV Feed",image)
                    cv2.setMouseCallback('OpenCV Feed', showBlank, i)

                if i[0]<1:
                    while True:
                        cv2.waitKey(10)
                        cv2.imshow("OpenCV Feed",image)
                        if i[0] >1 or i[0]==1:
                            break
                    frame_num=frame_num-1
                    continue

                #Apply collection logic
                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120,200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 4, cv2.LINE_AA)

                    cv2.putText(image, 'Collectiong frames for Video {}'.format(sequence), (15,12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 4, cv2.LINE_AA)
                    #show to screen
                    cv2.imshow("OpenCV Feed",image)
                    cv2.waitKey(1000)
                else:
                    cv2.putText(image, 'Collectiong frames for Video {}'.format(sequence), (15,12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 4,cv2.LINE_AA)

                #show to screen
                cv2.imshow("OpenCV Feed",image)

                #new export keypoints
                keypoints = extract_keypoints(results)
                npy_path=os.path.join(DATA_PATH, 'good', str(sequence),str(frame_num))
                np.save(npy_path,keypoints)
                badkeypoints1,badkeypoints2, badkeypoints3, badkeypoints4, badkeypoints5, badkeypoints6, badkeypoints7, badkeypoints8 = Hextract_badkeypoints(keypoints,x,y)
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
#crossentropy -> 수치로 표시하기에 유리한 방식으로 출력해주기 때문이다.
#왜 이러한 구조로 구성하였나? -> 
#1. 적은 양의 데이터만 사용할 예정이고
#2. 빠르게 학습시킬 수 있다는 장점과
#3. 실시간으로 평가를 빠르게 내려줄 수 있기 때문입니다.
def easy_train(text):
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
    model.fit(X_train,y_train,epochs=150,callbacks=[tb_callback])
    model.save('action.h5')

    sequence=[]
    sentence = []
    predictions=[]
    threshold = 0.4
    bad_pose_count=0
    start =0
    mp3file='good.mp3'
    settime=10
    i = [0]
    
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("OpenCV Feed", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("OpenCV Feed",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback("OpenCV Feed", showBlank, i )
    mode='mode0'
    #set mediapipe model
    with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:
        while cap.isOpened():
            #Read feed
            ret,frame = cap.read()
            if not ret:
                continue
            #make detection
            image , results  = mediapipe_detection(frame,holistic)
            #draw_landmark

            #mode1(스트레칭 모드), mode2, reset(초기화), end(끝내기)
            if i[0]!=5:
                print(i[0])
                if i[0]==-10:
                    mode='Mute'
                    settime=10
                elif i[0]==-20:
                    mode='Sound'
                    settime=60
                elif i[0]==-30:
                    mode='Reset'
                    settime=10
                    bad_pose_count=0
                elif i[0]==-40:
                    cap.release()
                    cv2.destroyAllWindows()
                    mode='End'

            #사용자의 모습이 보이지 않았을 경우 r키를 입력하여 다시 실행할 수 있게 한다.
            try:
                keypoints = extract_keypoints(results)
                sequence.insert(0,keypoints)
                sequence = sequence[:10]
                i[0]=5

            except Exception as e:
                i[0]=0
                cv2.putText(image,"Touch the Screen.",(100,100),cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,0,255), 4, cv2.LINE_AA)
                cv2.imshow("OpenCV Feed",image)
            
            if i[0]==0:
                while True:
                    cv2.waitKey(10)
                    cv2.putText(image,"Touch the Screen.",(100,100),cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,0,255), 4, cv2.LINE_AA)
                    cv2.imshow("OpenCV Feed",image)
                    bad_pose_count=0
                    if i[0] >1 or i[0]==1:
                        break
                continue

            else:
                now = datetime.datetime.now()
                cv2.putText(image,text,(1,25),cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,0), 4, cv2.LINE_AA)
                cv2.putText(image,"Mute",(120,25),cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,0), 4, cv2.LINE_AA)
                cv2.putText(image,"Sound",(239,25),cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,0), 4, cv2.LINE_AA)
                cv2.putText(image,"Reset",(375,25),cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,0), 4, cv2.LINE_AA)
                cv2.putText(image,"End",(511,25),cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,0), 4, cv2.LINE_AA)        

            if len(sequence) == 10:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                #3. vizs logic - 0.4보다 큰 수치를 가졌을 경우에 상태가 바뀌면 바뀐 상태로, 안바뀌면 안바뀐 상태로 
                if res[np.argmax(res)] > threshold:
                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                        #나쁜 자세가 1분 정도 지속적으로 유지되었을 때 음성으로 알려준다.
                        if actions[np.argmax(res)] !="good" and start == 0:
                            start = time.time()
                        elif actions[np.argmax(res)] == "good":
                            start = 0
                        elif actions[np.argmax(res)] !="good" :
                            dif=time.time()-start
                            if dif > settime:
                                start=time.time()
                                dif = 0
                                bad_pose_count=bad_pose_count+1
                                if mode=="Mute":
                                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                                else:
                                    pygame.mixer.music.load(mp3file)
                                    pygame.mixer.music.play()
                                cv2.imshow("OpenCV Feed",image)
                                cv2.waitKey(100)

                    else:
                        sentence.append(actions[np.argmax(res)])

                if len(sentence)>5:
                    sentence = sentence[-5:]

                #viz
                image = prob_viz(res,actions,image,colors,mode,bad_pose_count,now)
                #show to screen
                cv2.imshow("OpenCV Feed",image)

            #breaking
            if (cv2.waitKey(10) & 0xFF == ord('q')) or mode=='End':
                break

    cap.release()
    cv2.destroyAllWindows()



pygame.init()
pygame.display.set_caption('Desk_Trainer')
SURFACE = pygame.display.set_mode((1000,600))
FPSCLOCK = pygame.time.Clock()
Big_font = pygame.font.SysFont(None, 60)
studyImg = pygame.image.load("studying.png")
relaxImg = pygame.image.load("relax.png")
study_scale=pygame.transform.scale(studyImg,(170,160))
relax_scale=pygame.transform.scale(relaxImg,(170,160))
hardImg = pygame.image.load("hardmode.png")
easyImg = pygame.image.load("easymode.png")
hard_scale=pygame.transform.scale(hardImg,(160,150))
easy_scale=pygame.transform.scale(easyImg,(160,150))



class Button:
    def __init__(self, img_in, x, y, width, height, img_act, x_act, y_act, action = None):
        mouse = pygame.mouse.get_pos()
        click = pygame.mouse.get_pressed()
        flag="nothing"
        if x + width > mouse[0] > x and y + height > mouse[1] > y:
            SURFACE.blit(img_act,(x_act, y_act))
            if x_act ==263 and y_act==293 and flag=="nothing":
                flag='left'
            elif x_act ==613 and y_act==298 and flag=="nothing":
                flag="right"
            if click[0] and action == None and flag != "nothing":
                time.sleep(1)
                if flag=="left":
                    train(0.12,0.05)
                    easy_train("Hard")
                    flag="nothing"
                elif flag=="right":
                    train(0.18,0.1)
                    easy_train("Easy")
                    flag="nothing"
        else:
            SURFACE.blit(img_in,(x,y))
            
def main():
 
    
    message_caution = Big_font.render("Select Mode ", True, (0,0,0))
    while True:
        SURFACE.fill((255, 255, 255))
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            
        SURFACE.blit(message_caution, (360, 150))
        HardButton=Button(hard_scale,270,295,170,160,study_scale,263,293,None)
        easyButton=Button(easy_scale,620,300,170,160,relax_scale,613,298,None)
        pygame.display.update()
        FPSCLOCK.tick(30)


if __name__ == '__main__':
    main()