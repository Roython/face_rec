#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@file: face_recognition.py
@author: Roython
@Contact: rongyue_2014@163.com
@time: 19-1-6
'''
import cv2
import face_recognition
import numpy as np
import linecache
import time
from face_register import get_face_encoding
width,height = 768,432
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# fourcc = cv2.VideoWriter_fourcc(*'X264')
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
# fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
out = cv2.VideoWriter('./data/output.mp4',fourcc, 20.0, (width,height))


choos = input("weither to register face (y/n)")
if choos == 'y':
    get_face_encoding()
else:
    print('go next steps...')


recog=input("weither to recognize face (y/n):")
if recog=='y':
    face_file = "./data/face.txt"
    rec_name = ""
    rec_id_num = ""
    total_name = []
    total_id_num = []
    total_face_encoding = []
    font = cv2.FONT_HERSHEY_DUPLEX
    with open(face_file, 'r') as f:
        for i in range(len(linecache.getlines(face_file))):
            info_list=f.readline().split()
            name=info_list[0]
            id_num=info_list[1]
            face_encoding=info_list[2:]
            total_name.append(name)
            total_id_num.append(id_num)
            total_face_encoding.append(face_encoding)
    total_face_encoding=np.asarray(total_face_encoding).astype('float64')

    # print(total_name)
    # print(total_id_num)
    # print(total_face_encoding.dtype)
    # print(len(total_face_encoding[0]))

    # capture = cv2.VideoCapture(0)
    capture = cv2.VideoCapture('./data/test.mp4')
    while (capture.isOpened()):
        ret, frame = capture.read()
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        start = time.clock()

        face_locations = face_recognition.face_locations(rgb_small_frame, number_of_times_to_upsample=2, model='cnn')
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, num_jitters=10)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

            for i, v in enumerate(total_face_encoding):
                match = face_recognition.compare_faces([v], face_encoding, tolerance=0.5)

                rec_name = "Unknown"
                rec_id_num = "None"
                if match[0]:
                    rec_name = total_name[i]
                    rec_id_num = total_id_num[i]
                    break

            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, rec_name+'/'+rec_id_num, (left + 6, bottom - 6), font, 0.75,(255, 255, 255), 1)

        duration = time.clock() - start
        fps = 1 /duration

        cv2.putText(frame, 'FPS: '+str(round(fps,2)), (20,20), font, 0.75, (255, 0, 0), 1)
        cv2.imshow('Video', frame)
        out.write(frame)
        key = cv2.waitKey(1)
        if key & 0xFF==27:
            break


    capture.release()
    out.release()
    cv2.destroyAllWindows()

else:
    print('The End...')
