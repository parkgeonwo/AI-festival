import threading
from queue import Queue
import cv2
import mediapipe as mp
import numpy as np
import datetime
from PIL import ImageFont, ImageDraw, Image
import time
import requests
import keyboard


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

        if q2.qsize()==4:
            name_list.append(nowDatetime_path)

            cv2.imwrite("D:/ai-festival/practice/image/"+name_list[0]+".png", img)   # image save
            cv2.rectangle(img, (0,0),(resize_width,resize_height),(255,255,255),-1)  # shoot
            
            if q3.qsize()==0:
                q3.put((name_list[0]))

            # ## API 통신 + 결과 이미지 저장
            # files = {'photo': open('D:/ai-festival/practice/image/'+name_list[0]+'.png' , 'rb'), 'referral_id':'0000'}
            # response = requests.post('http://thenewme.m47rix.com/run', files=files)
            
            # try:
            #     data = response.json()     
            #     result_url = data['result_path']
            #     print(data)
            #     filename = result_url.split("/")[-1]         # url에서 맨뒤 "/" 뒤에 이름 가져오기

            #     save_img = Image.open(requests.get(result_url, stream = True).raw)    # 결과 url에서 image 받아서 save
            #     save_img.save('D:/ai-festival/practice/save_image/'+ filename)
                
            # except requests.exceptions.RequestException:
            #     print(response.text)
            
        if q2.qsize()==5:
            cv2.imshow(previewName, img)

        if q2.qsize()==6:
            upload_image = cv2.imread('D:/ai-festival/practice/save_image/after.png')
            img = cv2.addWeighted(img,0,upload_image,1,0)
            cv2.imshow(previewName, img)

        if q2.qsize()>=7:
          name_list=[]
          img = cv2.addWeighted(img,1,upload_image,0,0)

        cv2.imshow(previewName, img)

        if cv2.waitKey(5) & 0xFF == 27:
            break


def handler(q1,q2,q3):

    while True:
        time.sleep(0.01)

        if q1.qsize() != 0:      # v를 하고난 이후
            if q2.qsize()==0:
                q2.put((1))          # q2 size를 0 -> 1
            if q2.qsize() >=1 and q2.qsize() <= 3: # q2 size 1,2,3,4 -> countdown 3,2,1 + 찰칵
                time.sleep(1)
                q2.put((1))
            
            if q2.qsize()==4:
                time.sleep(1)
                q2.put((1))
            
            if q3.qsize()==1 and q2.qsize()==5:   # q3에 촬영된 이미지 이름이 들어오고, q2 size가 4일때(=촬영했을때)
                shoot_image_time = q3.get()
                # print(shoot_image_time)
                # q2.put((1))   # q2 size 5만들어주기
                ## API 통신 + 결과 이미지 저장
                files = {'photo': open('D:/ai-festival/practice/image/'+shoot_image_time+'.png' , 'rb'), 'referral_id':'0000'}
                response = requests.post('http://thenewme.m47rix.com/run', files=files)
                print(response)
                
                try:
                    data = response.json()     
                    result_url = data['result_path']
                    print(data)
                    filename = result_url.split("/")[-1]         # url에서 맨뒤 "/" 뒤에 이름 가져오기

                    save_img = Image.open(requests.get(result_url, stream = True).raw)    # 결과 url에서 image 받아서 save
                    save_img.save('D:/ai-festival/practice/save_image/'+ filename)

                    time.sleep(1)
                    q2.put((1))   # q2의 size가 6로 가기위해

                except requests.exceptions.RequestException:
                    print(response.text)
                    print("한번더 촬영해주세요")
                    q2.put((1))   # 오류나도 다음 스테이지로 넘어가기위해서
                
            if q2.qsize() == 6:  # 촬영후 사진 10초간 보여줌
                if keyboard.is_pressed("space"):
                    q2.put((1))

            if q2.qsize()>=7:  # q2 큐의 size가 5 이상이면
                time.sleep(2)   # 2초를 쉬고 q1,q2를 초기화
                q1.queue.clear()   # v를 안한상태로 되돌림
                q2.queue.clear() 


if __name__ == '__main__':
    q1 = Queue()
    q2 = Queue()
    q3 = Queue(1)

    thread1 = threading.Thread(target=cam, args=("Camera 0",0,q1,q2,q3,))
    thread2 = threading.Thread(target=handler, args=(q1,q2,q3,))

    thread1.start()
    thread2.start()













