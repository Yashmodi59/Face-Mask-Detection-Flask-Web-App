from email.message import EmailMessage

from keras.models import load_model
import numpy as np
import cv2
import tkinter
from tkinter import messagebox
import smtplib
import imghdr
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
import os
# Initialize Tkinter
root = tkinter.Tk()
root.withdraw()

# Load trained deep learning model
model = load_model('face_mask_detection_alert_system.h5')

# Classifier to detect face
face_det_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Capture Video
vid_source = cv2.VideoCapture(0)

# Dictionaries containing details of Wearing Mask and Color of rectangle around face. If wearing mask then color would be
# green and if not wearing mask then color of rectangle around face would be red
text_dict = {0: 'Mask ON', 1: 'No Mask'}
rect_color_dict = {0: (0, 255, 0), 1: (0, 0, 255)}

SUBJECT = "Subject"
TEXT = "One Visitor violated Face Mask Policy. See in the camera to recognize user. A Person has been detected without a face mask. Please Alert the authorities."
def SendMail(ImgFileName):
    img_data = open(ImgFileName, 'rb').read()
    msg = MIMEMultipart()
    msg['Subject'] = 'A person without mask detected'
    msg['From'] = 'ivrproject2020@gmail.com'
    msg['To'] = 'ivrproject2020@gmail.com'

    text = MIMEText("test")
    msg.attach(text)
    image = MIMEImage(img_data, name= os.path.basename(ImgFileName))
    msg.attach(image)

    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.ehlo()
    s.starttls()
    s.ehlo()
    s.login('ivrproject2020@gmail.com', '9898417832')
    s.sendmail(msg['From'], msg['To'], msg.as_string())
    s.quit()
# While Loop to continuously detect camera feed
while (True):

    ret, img = vid_source.read()
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(filename='saved_img.jpg', img=grayscale_img)

    faces = face_det_classifier.detectMultiScale(grayscale_img, 1.3, 5)

    for (x, y, w, h) in faces:

        face_img = grayscale_img[y:y + w, x:x + w]
        resized_img = cv2.resize(face_img, (112,112))
        normalized_img = resized_img / 255.0
        reshaped_img = np.reshape(normalized_img, (1, 112,112, 1))
        result = model.predict(reshaped_img)

        label = np.argmax(result, axis=1)[0]

        cv2.rectangle(img, (x, y), (x + w, y + h), rect_color_dict[label], 2)
        cv2.rectangle(img, (x, y - 40), (x + w, y), rect_color_dict[label], -1)
        cv2.putText(img, text_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # If label = 1 then it means wearing No Mask and 0 means wearing Mask
        if (label == 1):
            # Throw a Warning Message to tell user to wear a mask if not wearing one. This will stay
            # open and No Access will be given He/She wears the mask
            SendMail('saved_img.jpg')
            messagebox.showwarning("Warning", "Access Denied. Please wear a Face Mask")

        else:
            pass
            break

    cv2.imshow('LIVE Video Feed', img)
    key = cv2.waitKey(1)

    if (key == 27):
        break

cv2.destroyAllWindows()
source.release()
