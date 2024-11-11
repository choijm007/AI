import cv2
import face_recognition
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("./image/juwan2.jpg")
rgb_img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_encoding = face_recognition.face_encodings(rgb_img)[0]

img2 = cv2.imread("./image/group.jpg")
rgb_img2=cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img_encoding2 = face_recognition.face_encodings(rgb_img2)[0]

result=face_recognition.compare_faces([img_encoding], img_encoding2)
print("Result: ", result)

image = face_recognition.load_image_file("./image/jaemin1.jpg")
face_locations = face_recognition.face_locations(image)
plt.imshow(image)
print(face_locations)
for face_location in face_locations:
        plt.plot(face_location[1], face_location[0], 'bo')
        plt.plot(face_location[1], face_location[2], 'bo')
        plt.plot(face_location[3], face_location[2], 'bo')
        plt.plot(face_location[3], face_location[0], 'bo')

plt.show()

cv2.waitKey(0)


