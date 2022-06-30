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
    font = ImageFont.truetype('D:/ai-festival/practice/fonts/SCDream6.otf',40)

    # 촬영했을때 image가 두개 저장되는거 방지
    name_list = []

    max_num_hands = 10

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
    cv2.moveWindow(previewName, 1690,-35)

    # full screen 만들기
    # cv2.namedWindow(previewName, cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty(previewName,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    # cv2.moveWindow(previewName, 1705,0)

    cap = cv2.VideoCapture(camID)

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    resize_width, resize_height = 1290,730   # 2560, 1600  # 1920, 1440 # 2048,1280

    while cap.isOpened():   
        now = datetime.datetime.now()
        # nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
        nowDatetime_path = now.strftime('%Y-%m-%d-%H-%M-%S')      # 파일이름으로는 :를 못쓰기 때문에 따로

        ret, img = cap.read() 
        if not ret:     
            continue     
    
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (resize_width,resize_height))

        ### 화면 밝기 조절
        img = cv2.add(img,(60,60,60,0))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = hands.process(img)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks is not None:   
            for res in result.multi_hand_landmarks:   
                joint = np.zeros((21, 3))      
                for j, lm in enumerate(res.landmark):   
                    joint[j] = [lm.x, lm.y, lm.z]     

                v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:]
                v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] 
                v = v2 - v1 # [20,3]
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                angle = np.arccos(np.einsum('nt,nt->n',
                    v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                    v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) 

                angle = np.degrees(angle) 

                data = np.array([angle], dtype=np.float32) 
                ret, results, neighbours, dist = knn.findNearest(data, 3)     
                idx = int(results[0][0])   

                if idx == 9:
                    q1.put((nowDatetime_path))
                    # print(q1.qsize())

        ### 한글 넣기 (qsize 3이하)
        if q2.qsize() == 0:
            img = Image.fromarray(img)
            draw = ImageDraw.Draw(img)
            draw.text( xy=((resize_width/4)+50,30), text="V를 하면 AI가 캐릭터를 만듭니다.", font=font, fill=(0,0,0)  )
            img = np.array(img)

        # text_color = (0,255,255)

        ### q2의 size가 123(1초에 하나씩 늘어남)일때는 cnt에 3,2,1을 할당하고 화면에 표시
        for i in range(1,4):       
            if q2.qsize() == i:
                cnt = 4-i
                # cv2.putText(img, text=str(cnt), org=(int(resize_width*1/2),int(resize_height*1/2)),
                #     fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=4, color=text_color, thickness=5)
                
                img2 = cv2.imread("D:/ai-festival/practice/fonts/"+str(cnt)+"_resize.png")
                rows, cols, channels = img2.shape #로고파일 픽셀값 저장
                img_w, img_h, img2_w, img2_h = resize_width, resize_height, cols, rows

                # roi = img[50:rows+50,50:cols+50] #로고파일 필셀값을 관심영역(ROI)으로 저장함.
                roi = img[int(0.5*(img_h-img2_h)+0.2*img_h):int(0.5*(img_h+img2_h)+0.2*img_h),int(0.5*(img_w-img2_w)):int(0.5*(img_w+img2_w))] #로고파일 필셀값을 관심영역(ROI)으로 저장함.

                gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) #로고파일의 색상을 그레이로 변경
                ret, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY) #배경은 흰색으로, 그림을 검정색으로 변경
                mask_inv = cv2.bitwise_not(mask)
                
                # img_bg = cv2.bitwise_and(roi,roi,mask=mask) #배경에서만 연산 = img 배경 복사
                # img2_fg = cv2.bitwise_and(img2,img2, mask=mask_inv) #로고에서만 연산

                img_bg = cv2.bitwise_and(roi,roi,mask=mask_inv) #배경에서만 연산 = img 배경 복사
                img2_fg = cv2.bitwise_and(img2,img2, mask=mask) #로고에서만 연산
                
                # dst = cv2.bitwise_or(img_bg, img2_fg) #img_bg와 img2_fg를 합성
                dst = cv2.add(img_bg, img2_fg)
                
                # img[50:rows+50,50:cols+50] = dst #img에 dst값 합성
                img[int(0.5*(img_h-img2_h)+0.2*img_h):int(0.5*(img_h+img2_h)+0.2*img_h),int(0.5*(img_w-img2_w)):int(0.5*(img_w+img2_w))] = dst

        ### q2의 size가 4일때는 이미지 저장 및 q3에 저장된 이미지 이름 넣어주기
        if q2.qsize()==4:
            name_list.append(nowDatetime_path)
            cv2.imwrite("D:/ai-festival/practice/image/"+name_list[0]+".png", img)   # image save

            if q3.qsize()!=0:
                q3.queue.clear()
            if q3.qsize()==0:
                # print(name_list[0], "cam")
                q3.put((name_list[0]))  # q3에 찍힌 사진 이름 주기
                q2.put((1))       # q2 size를 5로
        
        ### q2 size가 5일때는 찰칵 이펙트
        if q2.qsize()==5:
            cv2.rectangle(img, (0,0),(resize_width,resize_height),(255,255,255),-1)  # shoot
            
        ### q2 size가 6일때는 그냥 웹캠
        if q2.qsize()==6:
            cv2.imshow(previewName, img)

        ### q2 size가 7일때는 API에서 받아온 이미지 노출
        if q2.qsize()==7:
            
            upload_image = cv2.imread('D:/ai-festival/practice/save_image/'+name_list[0]+'-after.png')

            if upload_image is None:
                upload_image = cv2.imread('D:/ai-festival/practice/save_image/error.png')
                upload_image = cv2.resize(upload_image, (resize_width,resize_height))

            img = cv2.addWeighted(img,0,upload_image,1,0)
            cv2.imshow(previewName, img)

        ### q2 size가 8일때는 다시웹캠, 모든 설정값 초기화
        if q2.qsize()>=8:
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
            if q2.qsize() >=1 and q2.qsize() <= 3: # q2 size 1,2,3,4 -> countdown 3,2,1
                time.sleep(1)
                q2.put((1))
            
            if q2.qsize()==4:     # 찰칵
                time.sleep(1)
            
            if q2.qsize()==5:
                time.sleep(0.5)
                q2.put((1))
            
            if q3.qsize()==1 and q2.qsize()==6:   # q3에 촬영된 이미지 이름이 들어오고, q2 size가 5일때(=촬영했을때)
                shoot_image_time = q3.get()
                # print(shoot_image_time,"handler")
                time.sleep(0.5)

                ## API 통신 + 결과 이미지 저장
                files = {'photo': open('D:/ai-festival/practice/image/'+shoot_image_time+'.png' , 'rb'), 'referral_id':'0000'}
                response = requests.post('http://thenewme.m47rix.com/run', files=files)
                
                try:
                    data = response.json()     
                    result_url = data['result_path']
                    print(data)
                    filename = result_url.split("/")[-1]         # url에서 맨뒤 "/" 뒤에 이름 가져오기

                    save_img = Image.open(requests.get(result_url, stream = True).raw)    # 결과 url에서 image 받아서 save
                    save_img.save('D:/ai-festival/practice/save_image/'+shoot_image_time+"-"+ filename)

                    time.sleep(1)
                    q2.put((1))   # q2의 size가 6로 가기위해

                except requests.exceptions.RequestException:
                    print(response.text)
                    print("한번더 촬영해주세요")
                    q2.put((1))   # 오류나도 다음 스테이지로 넘어가기위해서
                
            if q2.qsize() == 7:  # 촬영후 사진을 보여주는 단계
                if keyboard.is_pressed("space"): # space 누르면 q2 size 변경
                    q2.put((1))

            if q2.qsize()>=8:  # q2 큐의 size가 8 이상이면
                time.sleep(2)   # 2초를 쉬고 q1,q2를 초기화
                q1.queue.clear()   # v를 안한상태로 되돌림
                q2.queue.clear() 
            
            if keyboard.is_pressed("Esc"):
                break

        if keyboard.is_pressed("Esc"):
            break

if __name__ == '__main__':
    q1 = Queue()
    q2 = Queue()
    q3 = Queue(1)

    thread1 = threading.Thread(target=cam, args=("Camera",1,q1,q2,q3,))
    thread2 = threading.Thread(target=handler, args=(q1,q2,q3,))

    thread1.start()
    thread2.start()













