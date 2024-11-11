import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract
import re


class CheckCard:
    def __init__(self):
        self.birth = 24
        self.name = ''

    def checkNum(self, img, path, w, h):
        pytesseract.pytesseract.tesseract_cmd = path

        width1 = int(w / 2 - 318)
        width2 = int(w / 2 + 319)
        height1 = int(h / 2 - 206)
        height2 = int(h / 2 + 207)
        plt.figure(figsize=(12, 10))
        img_crop = img[height1:height2, width1:width2]
        # 노이즈 줄이기위해 블러 사용
        img2 = cv2.GaussianBlur(img_crop, ksize=(5, 5), sigmaX=0)

        # 임의의값 아래는 처리 안함,높으면 255로 고정
        img3 = cv2.adaptiveThreshold(img2, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)
        result = img3[150:210, 40:180]
        text = pytesseract.image_to_string(result)
        num = re.sub(r'[^0-9]', '', text)
        return num

    def checkName(self, img, path, w, h):
        pytesseract.pytesseract.tesseract_cmd = path

        width1 = int(w / 2 - 318)
        width2 = int(w / 2 + 319)
        height1 = int(h / 2 - 206)
        height2 = int(h / 2 + 207)
        plt.figure(figsize=(12, 10))
        img_crop = img[height1:height2, width1:width2]
        result = img_crop[80:160, 80:200]
        text = pytesseract.image_to_string(result, lang='kor')
        name = re.sub(r'[^가-힣]', '', text)
        return name

    def setBirth(self, path, cap):
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width1 = int(w / 2 - 318)
        width2 = int(w / 2 + 319)
        height1 = int(h / 2 - 206)
        height2 = int(h / 2 + 207)
        bool_Birth = False
        bool_Name = False
        birth_List = []
        name_List = []
        while True:
            ret, frame = cap.read()
            # 테두리 설정
            cv2.rectangle(frame, (width1, height1), (width2, height2), (0, 0, 255), 3)
            cv2.imshow('frame', frame)

            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))

            # 높이,너비,채널 저장
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if (not bool_Birth):
                agent_num = self.checkNum(img, path, w, h)
                if (len(agent_num) == 6):
                    birth_List.append(agent_num)
                    # print(agent_num) 숫자 확인용
                for i in birth_List:
                    if (birth_List.count(i) >= 3):
                        self.birth = i
                        bool_Birth = True
                        break

            if (not bool_Name):
                agent_name = self.checkName(img, path, w, h)
                if (len(agent_name) == 3):
                    name_List.append(agent_name)
                    # print(agent_name)
                for i in name_List:
                    if (name_List.count(i) >= 3):
                        self.name = i
                        bool_Name = True
                        break

            if cv2.waitKey(1) == ord('q'):
                cv2.imwrite("123456.jpg", img)
                return
            if (bool_Name and bool_Birth):
                return

    def getBirth(self):
        return self.birth

    def getName(self):
        return self.name

    def takePicture(self, cap, w, h):
        ret, img = cap.read()
        width1 = int(w / 2 - 318)
        width2 = int(w / 2 + 319)
        height1 = int(h / 2 - 206)
        height2 = int(h / 2 + 207)
        img_crop = img[height1:height2, width1:width2]
        user_img = img_crop[0:300, 350:660]
        return user_img

    def BirthCheck(self):
        a = int(self.birth[:2])
        if (30 < a < 99 or 0 <= a < 5):
            return True
        else:
            return False

    def run(self, path):
        cap = cv2.VideoCapture(0)
        height = 1600
        width = 1600
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)  # 가로
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)  # 세로
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.setBirth(path, cap)
        img = self.takePicture(cap, w, h)
        cap.release()
        cv2.destroyAllWindows()
        return img
