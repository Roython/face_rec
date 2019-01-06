import os
import cv2
import face_recognition


def get_face_encoding():
    name = input("please input your name: ")
    id_num = input("please input your id_num: ")
    file_dir = './data'
    file_name = 'face.txt'  # 存储人脸128向量的数据
    face_file = file_dir + '/' + file_name
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    if not os.path.exists(face_file):
        os.system(r"touch {}".format(face_file))  # 调用系统命令行来创建文件

    count=0
    # capture = cv2.VideoCapture(0)
    capture = cv2.VideoCapture('./data/face-demographics-walking-and-pause.mp4')
    print("Now swing your head to a certain position and press the s key to save face encoding.")

    while (capture.isOpened()):
        ret, frame = capture.read()
        print(frame.shape)

        top_list, right_list, bottom_list, left_list, face_area_list = [], [], [], [], []

        key = cv2.waitKey(1)

        face_locations = face_recognition.face_locations(frame,2,'cnn')

        if len(face_locations) and (key & 0xFF == ord('s')):

            for face_location in face_locations:
                top, right, bottom, left = face_location
                face_area = (bottom-top)*(right-left)
                top_list.append(top); right_list.append(right); bottom_list.append(bottom); left_list.append(left); face_area_list.append(face_area)

            index = (face_area_list.index(max(face_area_list)))
            image = frame[top_list[index] - 40:bottom_list[index] + 40, left_list[index] - 40:right_list[index] + 40]
            cv2.imshow('output', image)
            # face_location[index] = dlib.rectangle(top_list[index], right_list[index], bottom_list[index], left_list[index])
            face_encoding = face_recognition.face_encodings(image, num_jitters=10)
            # print(face_encoding[0])
            save_data_to_file(face_file, name, id_num, face_encoding[0], count)

            cv2.rectangle(frame, (left_list[index], top_list[index] - 20), (right_list[index], bottom_list[index] + 20), (255, 0, 0), 2)
            cv2.putText(frame, "count: %d" % count, (left_list[index] - 10, top_list[index] - 25), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 3)
            count += 1

        cv2.imshow('Video', frame)
        if key & 0xFF ==27 or count == 50:
            break

    capture.release()
    cv2.destroyAllWindows()


def save_data_to_file(file, name, id_num, face_array, num):
    with open(file, 'a+') as f:
        f.write(name+' '+id_num+' ')
        for i in range(127):
            f.write(str(face_array[i])+' ')
        f.write(str(face_array[127])+'\n')
    print("save {} face encoding done !!!".format(num))


if __name__=="__main__":
    get_face_encoding()
