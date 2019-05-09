import PySimpleGUI as sg
import numpy as np
import cv2
import pickle
import string
import random
import os

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

labels = {"person_name" : 1}
with open("labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)

def random_generator(size=10, chars=string.ascii_uppercase + string.digits):
	return ''.join(random.choice(chars) for _ in range(size))

image_count = 0;
while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
	for (x, y, w, h) in faces: 
		# print(x,y,w,h)
		roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
		roi_color = frame[y:y+h, x:x+w]

		if not os.path.exists('images/new'):
			os.makedirs('images/new')
		image_name = random_generator() + ".png"
		img_item = "images/new/" + image_name
		# cv2.imwrite(img_item, roi_color) #Save face to file
		cv2.imwrite(img_item, frame) #Save face to file
		image_count += 1
		if image_count == 20:
			# Close the camera view
			cap.release()
			cv2.destroyAllWindows()

			layout = [
				[sg.Text('Enter your name')],
				[sg.InputText()],
				[sg.Submit(), sg.Cancel()]
			]

			event, values  = sg.Window('Everything bagel', layout, auto_size_text=True, default_element_size=(40, 1)).Read()
			if event == 'Submit':
				os.rename('images/new', 'images/' +values[0])
				os.makedirs('images/new')
				os.system('python3 faces-train.py')

	# Display the resulting frame
	cv2.imshow('frame',frame)
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()