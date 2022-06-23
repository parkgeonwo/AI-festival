####################### 한손 ########################3

# import cv2
# import mediapipe as mp
# import numpy as np

# max_num_hands = 1

# rps_gesture = {0:'rock', 5:'paper', 9:'scissors'}      # 가위바위보를 하기 위해 0,5,9번을 새롭게 정의

# # MediaPipe hands model   # mediapipe 손 인식
# mp_hands = mp.solutions.hands        
# mp_drawing = mp.solutions.drawing_utils    # mediapipe의 drawing_utils 웹캠영상에서 손가락의 뼈마디 부분을 그릴 수 있도록 도와주는 유틸리티

# hands = mp_hands.Hands(               # 손가락 detection 모듈 초기화
#     max_num_hands=max_num_hands,       # 최대 몇개의 손을 인식할것인지 
#     min_detection_confidence=0.5,        # 0.5로 하는게 가장 좋다.
#     min_tracking_confidence=0.5)

# # Gesture recognition model  # 제스쳐 인식 모델
# file = np.genfromtxt('D:/ai-festival/practice/data/gesture_train.csv', delimiter=',')    # data에 gesture_train.csv파일이 있다. 각각의 제스쳐/각도/라벨 저장되어있음
# # file = np.genfromtxt('./data/gesture_train.csv', delimiter=',')

# angle = file[:,:-1].astype(np.float32)        # 앵글 데이터를 모아줌
# label = file[:, -1].astype(np.float32)        # 라벨 데이터를 모아줌
# knn = cv2.ml.KNearest_create()                # opencv의 k-nearst-neighbors 알고리즘 사용하여
# knn.train(angle, cv2.ml.ROW_SAMPLE, label)      # 학습을 시켜버림

# cap = cv2.VideoCapture(0) # 웹캠을 열어야 하는데 opencv의 video capture를 사용해서 웹캠을 열어준다.
# # 카메라를 여러개 쓰는 사람은 0번말고 숫자를 조정해주면 된다.

# while cap.isOpened():   # opencv의 비디오캡쳐를 initialize했던걸 한개의 프레임마다 읽어올것이다. # 카메라가 열려 있으면 loop
#     ret, img = cap.read() # 한 프레임씩 읽어옴 / 읽는데 성공했다면 --> img = cv2.flip(img, 1) 아래로 실행
#     if not ret:     # 성공하지 못했다면, 
#         continue     # 다음프레임으로 넘어감 

#     # mediapipe에 넣기전에 전처리를 해줘야 한다.
#     # opencv는 BGR / mediapipe는 RGB 컬러 시스템이기 때문에 cvtColor를 사용해서 opencv로 읽어온 한 프레임을 bgr -> rgb로 변경해주고,
#     # 만약 거울형태로 뒤집어 있으면 flip을 해줘야한다.
#     img = cv2.flip(img, 1)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     # mediapipe에 넣어주기전에 전처리를 해줘야하는데, hands.process에 image를 넣어주면 전처리된 이미지가 result에 들어간다.
#     result = hands.process(img)

#     # 이미지를 출력해야하니까 다시 rgb -> bgr로 변경
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

#     # 위 과정을 통해 전처리가 되고, 모델까지 inference 되고, result까지 나오게 되면
#     if result.multi_hand_landmarks is not None:   # result에 뭔가 들어있다면 = 손을 인식했다면
#         for res in result.multi_hand_landmarks:   # 여러개의 손이 있을 수 있으니까 for문을 통해 반복을 돌고
#             joint = np.zeros((21, 3))      # 각각의 빨간 점들을 joint라고 함. 이게 21개이고 xyz좌표 3개
#             for j, lm in enumerate(res.landmark):   # 각 joint마다 landmark를 저장하는데
#                 joint[j] = [lm.x, lm.y, lm.z]     # xyz 좌표를 joint를 저장해준다.  --> 21,3 array가 생성된다.

#             # Compute angles between joints
#             # 각 joint로 벡터를 계산해서 각도를 계산 -> 각각 관절(joint와 joint사이)에 대한 벡터를 구해줌 
#             v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
#             v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
#             v = v2 - v1 # [20,3]
#             # Normalize v
#             # 벡터를 각각 길이로 나눠줘서 normalize 해줌 --> 단위벡터(크기 1)
#             v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

#             # Get angle using arcos of dot product
#             # 벡터의 내적을 이용해서 각도를 구해준다. 크기가 1이므로 각각의 내적값은 두벡터가 이루는 cos값이 되고
#             # 이를 역함수인 arccos를 넣어주면 각각이 이루는 각을 계산해준다.
#             angle = np.arccos(np.einsum('nt,nt->n',
#                 v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
#                 v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]  # 15개의 각도를 구해서 angle변수에 저장

#             angle = np.degrees(angle) # Convert radian to degree # 라디안 값을 각으로 변형

#             # 아까 제스쳐모델을 학습시켰는데, 학습시킨 knn 모델을 가져다가 inference를 진행한다.
#             # Inference gesture
#             data = np.array([angle], dtype=np.float32) # numpy array로 바꿔주고, float32비트로 형태로 바꿔주고 
#             ret, results, neighbours, dist = knn.findNearest(data, 3)     # k가 3일때의 값을 구해줌
#             idx = int(results[0][0])    # 결과는 result의 첫번째 index에 저장

#             # Draw gesture result
#             if idx in rps_gesture.keys():    # 만약에 인덱스가 가위바위보중에 하나라면,
#                 cv2.putText(img, text=rps_gesture[idx].upper(), org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
#                 # 가위바위보 결과를 표시한다.
#                 # opencv의 putText를 사용해서 가위인지,바위인지,보인지 글씨로 쓰도록 함. 

#             # Other gestures    # 모든 제스쳐를 활용하고 싶다면 아래코드를 사용
#             # cv2.putText(img, text=gesture[idx].upper(), org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

#             mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
#             # 손가락 마디마디에 랜드마크를 그리는 함수를 미디어 파이프의 drawrandmark 함수를 이용해서 그린다.

#     # 그림을 실제로 보여줌
#     cv2.imshow('Game', img)

#     if cv2.waitKey(5) & 0xFF == 27:
#       break













################################# 두손 #######################################3
# import cv2
# import mediapipe as mp
# import numpy as np

# max_num_hands = 2       # 두개의 손 인식
# gesture = {
#     0:'fist', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
#     6:'six', 7:'rock', 8:'spiderman', 9:'yeah', 10:'ok',
# }
# rps_gesture = {0:'rock', 5:'paper', 9:'scissors'}

# # MediaPipe hands model
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# hands = mp_hands.Hands(
#     max_num_hands=max_num_hands,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5)

# # Gesture recognition model
# file = np.genfromtxt('D:/ai-festival/practice/data/gesture_train.csv', delimiter=',')
# angle = file[:,:-1].astype(np.float32)
# label = file[:, -1].astype(np.float32)
# knn = cv2.ml.KNearest_create()
# knn.train(angle, cv2.ml.ROW_SAMPLE, label)

# cap = cv2.VideoCapture(0)

# while cap.isOpened():
#     ret, img = cap.read()
#     if not ret:
#         continue

#     img = cv2.flip(img, 1)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     result = hands.process(img)

#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

#     if result.multi_hand_landmarks is not None:
#         rps_result = []        # 가위바위보한 결과 & 손의 좌표를 저장한다.

#         for res in result.multi_hand_landmarks:
#             joint = np.zeros((21, 3))
#             for j, lm in enumerate(res.landmark):
#                 joint[j] = [lm.x, lm.y, lm.z]

#             # Compute angles between joints
#             v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
#             v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
#             v = v2 - v1 # [20,3]
#             # Normalize v
#             v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

#             # Get angle using arcos of dot product
#             angle = np.arccos(np.einsum('nt,nt->n',
#                 v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
#                 v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

#             angle = np.degrees(angle) # Convert radian to degree

#             # Inference gesture
#             data = np.array([angle], dtype=np.float32)
#             ret, results, neighbours, dist = knn.findNearest(data, 3)
#             idx = int(results[0][0])

#             # Draw gesture result
#             if idx in rps_gesture.keys():
#                 org = (int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0]))
#                 cv2.putText(img, text=rps_gesture[idx].upper(), org=(org[0], org[1] + 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

#                 rps_result.append({            # rps_result에다가
#                     'rps': rps_gesture[idx],     # gesture결과를(가위,바위,보) 저장해두고
#                     'org': org    # 각 손의(두개의 손의) 위치를 저장해준다.
#                 })

#             mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

#             print(rps_result)

#             # Who wins?
#             if len(rps_result) >= 2:  # 손이 두개이상이면 아래코드를 실행
#                 winner = None
#                 text = ''

#                 if rps_result[0]['rps']=='rock':     # 첫번째 사람이 바위를 냈을때,
#                     if rps_result[1]['rps']=='rock'     : text = 'Tie'     # 두번째사람도 바위를 내면 비겼다(tie).
#                     elif rps_result[1]['rps']=='paper'  : text = 'Paper wins'  ; winner = 1  # 두번째사람(1번사람)이 보를 내면 1번사람이 이겼다.
#                     elif rps_result[1]['rps']=='scissors': text = 'Rock wins'   ; winner = 0  # 가위를 내면 1번사람이 졌다.
#                 elif rps_result[0]['rps']=='paper':
#                     if rps_result[1]['rps']=='rock'     : text = 'Paper wins'  ; winner = 0
#                     elif rps_result[1]['rps']=='paper'  : text = 'Tie'
#                     elif rps_result[1]['rps']=='scissors': text = 'Scissors wins'; winner = 1
#                 elif rps_result[0]['rps']=='scissors':
#                     if rps_result[1]['rps']=='rock'     : text = 'Rock wins'   ; winner = 1
#                     elif rps_result[1]['rps']=='paper'  : text = 'Scissors wins'; winner = 0
#                     elif rps_result[1]['rps']=='scissors': text = 'Tie'

#                 # 누가이겼는지 글씨로 표시해주는 코드
#                 if winner is not None:
#                     cv2.putText(img, text='Winner', org=(rps_result[winner]['org'][0], rps_result[winner]['org'][1] + 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 255, 0), thickness=3)
#                 cv2.putText(img, text=text, org=(int(img.shape[1] / 2), 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255), thickness=3)

#     cv2.imshow('Game', img)
#     if cv2.waitKey(1) == ord('q'):
#         break



############################ cam 2개 키기 ######################
# import cv2
# import threading

# class camThread(threading.Thread):
#     def __init__(self, previewName, camID):
#         threading.Thread.__init__(self)
#         self.previewName = previewName
#         self.camID = camID
#     def run(self):
#         print("Starting " + self.previewName)
#         camPreview(self.previewName, self.camID)

# def camPreview(previewName, camID):
#     cv2.namedWindow(previewName)
#     cam = cv2.VideoCapture(camID)
#     if cam.isOpened():  # try to get the first frame
#         rval, frame = cam.read()
#     else:
#         rval = False

#     while rval:
#         cv2.imshow(previewName, frame)
#         rval, frame = cam.read()
#         key = cv2.waitKey(20)
#         if key == 27:  # exit on ESC
#             break
#     cv2.destroyWindow(previewName)

# # Create two threads as follows
# thread1 = camThread("Camera 1", 0)
# thread2 = camThread("Camera 2", 1)
# thread1.start()
# thread2.start()



####################### hand detection #########################

# import cv2
# import mediapipe as mp
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# mp_hands = mp.solutions.hands

# cap = cv2.VideoCapture(1)
# width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# # cap.set(cv2.CAP_PROP_FRAME_WIDTH,2048)
# # cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1152)

# with mp_hands.Hands(
#     model_complexity=0,
#     max_num_hands=1,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5) as hands:
#   while cap.isOpened():
#     success, image = cap.read()
#     # image = cv2.resize(image, (2048,1152))
#     image = cv2.resize(image, (2560,1440), interpolation=cv2.INTER_LANCZOS4)
#     # image = cv2.resize(image, (3072,1728))
#     if not success:
#       print("Ignoring empty camera frame.")
#       continue

#     image.flags.writeable = False
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = hands.process(image)

#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     if results.multi_hand_landmarks:
#       for hand_landmarks in results.multi_hand_landmarks:
#         mp_drawing.draw_landmarks(
#             image,
#             hand_landmarks,
#             mp_hands.HAND_CONNECTIONS,
#             mp_drawing_styles.get_default_hand_landmarks_style(),
#             mp_drawing_styles.get_default_hand_connections_style())
        
#         # xlist = []
#         # ylist = []
#         # landmark_list = []
#         # for id, lm in enumerate(hand_landmarks.landmark):
#         #   x,y = lm.x*width, lm.y*height
#         #   landmark_list.append([x,y])

#     cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
#     if cv2.waitKey(5) & 0xFF == 27:
#       break
# cap.release()



######### 이미지 불러오기 ############3

# import cv2

# img = cv2.imread("D:/ai-festival/practice/image/2022-06-22-11-38-46.jpg")
# print(img)    # numpy array


# cv2.imshow("image", img)
# cv2.waitKey()
# cv2.destroyAllWindows()


############# 한글 적기 ################


# from PIL import ImageFont, ImageDraw, Image
# import cv2
# import numpy as np

# font = ImageFont.truetype('D:/ai-festival/practice/fonts/SCDream6.otf',30)

# cap = cv2.VideoCapture(0)

# while(True):
#     ret, frame = cap.read()    # Read 결과와 frame
#     if(ret) :

#         frame = Image.fromarray(frame)
#         draw = ImageDraw.Draw(frame)
#         draw.text( xy=(30,20), text="V를 하면 사진이 찍힙니다.", font=font, fill=(0,0,0)  )
#         frame = np.array(frame)

#         cv2.imshow('Window', frame)

#         if cv2.waitKey(10) == ord('q'):
#             break

# cap.release()
# cv2.destroyAllWindows()




################ 이미지 합성하기 ################


# from PIL import ImageFont, ImageDraw, Image
# import cv2
# import numpy as np

# font = ImageFont.truetype('D:/ai-festival/practice/fonts/SCDream6.otf',30)

# cap = cv2.VideoCapture(0)
# width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

# while(True):
#     ret, img = cap.read()    # Read 결과와 img
#     if(ret) :

#         img = Image.fromarray(img)
#         draw = ImageDraw.Draw(img)
#         draw.text( xy=(30,20), text="V를 하면 사진이 찍힙니다.", font=font, fill=(0,0,0)  )
#         img = np.array(img)

#         upload_image = cv2.imread('D:/ai-festival/practice/save_image/after2.png')
#         rows, cols, channels = upload_image.shape
#         roi = img[50:rows+50,50:cols+50]

#         gray = cv2.cvtColor(upload_image, cv2.COLOR_BGR2GRAY) #로고파일의 색상을 그레이로 변경
#         ret, mask = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY) #배경은 흰색으로, 그림을 검정색으로 변경
#         mask_inv = cv2.bitwise_not(mask)
#         # cv2.imshow('mask',mask) #배경 흰색, 로고 검정
#         # cv2.imshow('mask_inv',mask_inv) # 배경 검정, 로고 흰색
        
#         src1_bg = cv2.bitwise_and(roi,roi,mask=mask) #배경에서만 연산 = src1 배경 복사
#         # cv2.imshow('src1_bg',src1_bg)
        
#         src2_fg = cv2.bitwise_and(upload_image, upload_image, mask = mask_inv) #로고에서만 연산
#         # cv2.imshow('src2_fg',src2_fg)
        
#         dst = cv2.bitwise_or(src1_bg, src2_fg) #src1_bg와 src2_fg를 합성
#         # cv2.imshow('dst',dst)
        
#         img[50:rows+50,50:cols+50] = dst #src1에 dst값 합성

#         cv2.imshow('Window', img)

#         if cv2.waitKey(5) & 0xFF == 27:
#             break

# cap.release()
# cv2.destroyAllWindows()






# from PIL import ImageFont, ImageDraw, Image
# import cv2
# import numpy as np
# import time

# font = ImageFont.truetype('D:/ai-festival/practice/fonts/SCDream6.otf',30)

# cap = cv2.VideoCapture(0)
# width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

# while(True):
#     ret, img = cap.read()    # Read 결과와 img
#     if(ret) :

#         img = Image.fromarray(img)
#         draw = ImageDraw.Draw(img)
#         draw.text( xy=(30,20), text="V를 하면 사진이 찍힙니다.", font=font, fill=(0,0,0)  )
#         img = np.array(img)

#         upload_image = cv2.imread('D:/ai-festival/practice/save_image/after.png')
#         img = cv2.addWeighted(img,0,upload_image,1,0)

#         time.sleep(3)

#         img = cv2.addWeighted(img,1,upload_image,0,0)

#         cv2.imshow('Window', img)

#         if cv2.waitKey(5) & 0xFF == 27:
#             break

# cap.release()
# cv2.destroyAllWindows()





########################### 테스트 #######################

from distutils.command.upload import upload
import threading
from queue import Queue
import cv2
import mediapipe as mp
import numpy as np
import datetime
from PIL import ImageFont, ImageDraw, Image
import time
import requests


def cam(previewName, camID,q1,q2,q3,):

    # font
    # font = ImageFont.truetype('./fonts/SCDream6.otf',20)
    font = ImageFont.truetype('D:/ai-festival/practice/fonts/SCDream6.otf',30)

    # 촬영했을때 image가 두개 저장되는거 방지
    name_list = []

    max_num_hands = 1

    mp_hands = mp.solutions.hands        
    # mp_drawing = mp.solutions.drawing_utils

    hands = mp_hands.Hands(               
        max_num_hands=max_num_hands,       
        min_detection_confidence=0.5,        
        min_tracking_confidence=0.5)

    # Gesture recognition model  # 제스쳐 인식 모델
    file = np.genfromtxt('D:/ai-festival/practice/data/gesture_train.csv', delimiter=',')    # data에 gesture_train.csv파일이 있다. 각각의 제스쳐/각도/라벨 저장되어있음
    # file = np.genfromtxt('./data/gesture_train.csv', delimiter=',')

    angle = file[:,:-1].astype(np.float32)        # 앵글 데이터를 모아줌
    label = file[:, -1].astype(np.float32)        # 라벨 데이터를 모아줌
    knn = cv2.ml.KNearest_create()                # opencv의 k-nearst-neighbors 알고리즘 사용하여
    knn.train(angle, cv2.ml.ROW_SAMPLE, label)      # 학습을 시켜버림

    cv2.namedWindow(previewName)
    cap = cv2.VideoCapture(camID)

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # print(width,height)

    while cap.isOpened():   
        now = datetime.datetime.now()
        # nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
        nowDatetime_path = now.strftime('%Y-%m-%d-%H-%M-%S')      # 파일이름으로는 :를 못쓰기 때문에 따로

        ret, img = cap.read() 
        if not ret:     
            continue     

        img = cv2.flip(img, 1)

        resize_width, resize_height = 640,480     # 2560, 1600  # 1920, 1440
        img = cv2.resize(img, (resize_width,resize_height))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = hands.process(img)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks is not None:   
            for res in result.multi_hand_landmarks:   
                joint = np.zeros((21, 3))      
                for j, lm in enumerate(res.landmark):   
                    joint[j] = [lm.x, lm.y, lm.z]     

                # 각 joint로 벡터를 계산해서 각도를 계산 -> 각각 관절(joint와 joint사이)에 대한 벡터를 구해줌 
                v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
                v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
                v = v2 - v1 # [20,3]
                # 벡터를 각각 길이로 나눠줘서 normalize 해줌 --> 단위벡터(크기 1)
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # 벡터의 내적을 이용해서 각도를 구해준다. 크기가 1이므로 각각의 내적값은 두벡터가 이루는 cos값이 되고
                # 이를 역함수인 arccos를 넣어주면 각각이 이루는 각을 계산해준다.
                angle = np.arccos(np.einsum('nt,nt->n',
                    v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                    v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]  # 15개의 각도를 구해서 angle변수에 저장

                angle = np.degrees(angle) # Convert radian to degree # 라디안 값을 각으로 변형

                # 아까 제스쳐모델을 학습시켰는데, 학습시킨 knn 모델을 가져다가 inference를 진행한다.  # Inference gesture
                data = np.array([angle], dtype=np.float32) # numpy array로 바꿔주고, float32비트로 형태로 바꿔주고 
                ret, results, neighbours, dist = knn.findNearest(data, 3)     # k가 3일때의 값을 구해줌
                idx = int(results[0][0])    # 결과는 result의 첫번째 index에 저장

                if idx == 9:
                    q1.put((nowDatetime_path))
                    # print(q1.qsize())

        ### 한글 넣기
        if q2.qsize() <= 3:
            img = Image.fromarray(img)
            draw = ImageDraw.Draw(img)
            draw.text( xy=(30,20), text="V를 하면 사진이 찍힙니다.", font=font, fill=(0,0,0)  )
            img = np.array(img)

        text_color = (0,255,255)

        for i in range(1,4):      # q2의 size가 123(1초에 하나씩 늘어남)일때는 cnt에 3,2,1을 할당하고 화면에 표시 
            if q2.qsize() == i:
                cnt = 4-i
                cv2.putText(img, text=str(cnt), org=(int(resize_width*1/2),int(resize_height*1/2)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=4, color=text_color, thickness=5)

        if q2.qsize() == 4:
            name_list.append(nowDatetime_path)
            q3.put((name_list[0]))

            # cv2.imwrite("./image/"+nowDatetime_path+".png", img)
            cv2.imwrite("D:/ai-festival/practice/image/"+name_list[0]+".png", img)   # image save
            cv2.rectangle(img, (0,0),(resize_width,resize_height),(255,255,255),-1)  # shoot

            ## API 통신 + 결과 이미지 저장
            files = {'photo': open('D:/ai-festival/practice/image/'+name_list[0]+'.png' , 'rb'), 'referral_id':'0000'}
            print(files)
            response = requests.post('http://thenewme.m47rix.com/run', files=files)
            
            try:
                data = response.json()     
                result_url = data['result_path']
                print(data)
                filename = result_url.split("/")[-1]         # url에서 맨뒤 "/" 뒤에 이름 가져오기

                save_img = Image.open(requests.get(result_url, stream = True).raw)    # 결과 url에서 image 받아서 save
                save_img.save('D:/ai-festival/practice/save_image/'+ filename)
                
            except requests.exceptions.RequestException:
                print(response.text)
            
        if q2.qsize()==5:
            upload_image = cv2.imread('D:/ai-festival/practice/save_image/after.png')
            print(upload_image.shape)
            img = cv2.addWeighted(img,0,upload_image,1,0)
            cv2.imshow(previewName, img)
        else:
            cv2.imshow(previewName, img)

        if q2.qsize()>=6:
          name_list=[]
          img = cv2.addWeighted(img,1,upload_image,0,0)


        if cv2.waitKey(5) & 0xFF == 27:
            break


def handler(q1,q2,q3):

    while True:
        time.sleep(0.1)
        print(q2.qsize())

        if q1.qsize() != 0:      # v를 했을때
            q2.put((1))          # q2 에 put
            if q2.qsize() >=1 and q2.qsize() <= 4:
                time.sleep(1)
            if q2.qsize() == 5:  # 촬영후 사진 10초간 보여줌
                time.sleep(10)
            if q2.qsize()>=6:  # q2 큐의 size가 5 이상이면
                time.sleep(2)   # 2초를 쉬고 q1,q2를 초기화
                q1.queue.clear()
                q2.queue.clear()


if __name__ == '__main__':
    q1 = Queue()
    q2 = Queue()
    q3 = Queue(1)

    thread1 = threading.Thread(target=cam, args=("Camera 0",0,q1,q2,q3,))
    thread2 = threading.Thread(target=handler, args=(q1,q2,q3,))

    thread1.start()
    thread2.start()



################ open rb test #########################3

# file = open('D:/ai-festival/practice/image/2022-06-23-14-21-26.png' , 'rb') 
# print(file)  # <_io.BufferedReader name='D:/ai-festival/practice/image/2022-06-23-14-21-26.png'>
# file.close()


























