#!/usr/bin/python

# Import the required modules
import cv2, os
import picamera
import numpy as np
from PIL import Image
import RPi.GPIO as GPIO
from PIL import Image, ImageEnhance
import requests
import twilio
import twilio.rest
from twilio.rest import Client

account_sid = "ACb6d41c1ff66824c4ab083e23579cf6a0"
auth_token = "596e7d39ca31ae0e4a6c3429129957e4"
client = Client(account_sid, auth_token)
url='http://35.167.121.118/student_attendance/post.php?control='

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(3,GPIO.IN)
i=GPIO.input(3)

# For face detection we will use the Haar Cascade provided by OpenCV.
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)



# For face recognition we will the the LBPH Face Recognizer 
recognizer = cv2.createLBPHFaceRecognizer()

def get_images_and_labels(path):
    # Append all the absolute image paths in a list image_paths
    # We will not read the image with the .sad extension in the training set
    # Rather, we will use them to test our accuracy of the training
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.endswith('.')]
    # images will contains face images
    images = []
    # labels will contains the label that is assigned to the image
    labels = []
    var = 0
    i=1
    for image_path in image_paths:
        # Read the image and convert to grayscale
        
        image_path=('/home/pi/Desktop/face/db2/student%s.jpg' % i)
        i=i+1
        print(image_path)
        image_pil = cv2.imread(image_path)
        gray = cv2.cvtColor(image_pil,cv2.COLOR_BGR2GRAY)
        # Convert the image format into numpy array
        # image_pil = ImageEnhance.Brightness(image_pil)
        image = np.array(gray, 'uint8')
        #image = np.array(image_pil)
        img=cv2.resize(image,(128,128))
        
        # Get the label of the image
        #nbr = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
        # Detect the face in the image
        faces = faceCascade.detectMultiScale(img)
        var=var+1
        
        # If face is detected, append the face to images and the label to labels
        for (x, y, w, h) in faces:
            images.append(img[y:y+h,x:x+w])
            labels.append(var)
            cv2.imshow("Adding faces to traning set...", img[y:y+h,x:x+w])
            cv2.waitKey(50)
            
    # return the images list and labels list
    
    print(images)
    return images, labels

# Path to the Yale Dataset
path = './db2'

images, labels = get_images_and_labels(path)
cv2.destroyAllWindows()
print(labels)

print(np.array(labels))
recognizer.train(images,np.array(labels))
while True:
    if i==1:
        with picamera.PiCamera() as cam:
             cam.capture('image.jpg')
        predict_image_pil = cv2.imread('image.jpg')
        gray = cv2.cvtColor(predict_image_pil,cv2.COLOR_BGR2GRAY)
        predict_image = np.array(gray, 'uint8')
        #predict_image = np.array(predict_image_pil)
        predict_image =cv2.resize(predict_image,(128,128))
        # predict_image = ImageEnhance.Sharpness(predict_image)
        faces = faceCascade.detectMultiScale(predict_image)
        for (x, y, w, h) in faces:
            predicted,conf= recognizer.predict(predict_image[y:y+h,x:x+w])
            cv2.imshow("Adding faces to traning set...", predict_image[y:y+h,x:x+w])
            cv2.waitKey(50)
        #print(predicted, conf)
        j=0
        k=0
        l=0
        if(predicted>=1 and predicted<=8):
            j=1
            print('ID1')
            url='http://35.167.121.118/student_attendance/post.php?control=12'
            r=requests.post(url)
            print(r)
            
        if (predicted>=9 and predicted<=16):
            k=1
            print('ID2')
            url='http://35.167.121.118/student_attendance/post.php?control=05'
            r=requests.post(url)
            print(r)
            
        if(predicted>16):
            l=1
            print('ID3')
            url='http://35.167.121.118/student_attendance/post.php?control=33'
            r=requests.post(url)
            print(r)
        
        if(j==0):
            print('ID1 absent')
            #message = client.api.account.messages.create(to="+919986551632", from_="+15807814327",
                                     #body="ID1 Absent!")
            print('Message Sent Successfully')
        if(k==0):
            print('ID2 absent')
            #message = client.api.account.messages.create(to="+919986551632", from_="+15807814327",
                                     #body="ID2 Absent!")
            print('Message Sent Successfully')
        if l==0:
            print('ID-3 absent')
            #message = client.api.account.messages.create(to="+919986551632", from_="+15807814327",
                                     #body="ID3 Absent!")
            print('Message Sent Successfully')
            

