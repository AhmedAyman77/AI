import cv2
import numpy as np
import joblib
import json
import base64
import os
from wavelet import w2d

__class_name_to_number = {}
__class_number_to_name = {}
__model = None


def number_2_name(class_num):
    return __class_number_to_name[class_num]

def classify_image(base64_data, file_path=None):

    # get and prepare image
    imgs = get_face_if_2_eyes(file_path,base64_data)
    result = []

    for img in imgs:
        o_img = cv2.resize(img, (32, 32))
        w_img = w2d(img, 'db1', 5)
        w_img = cv2.resize(w_img, (32, 32))
        combined_img = np.vstack((o_img.reshape(32 * 32 * 3, 1), w_img.reshape(32 * 32, 1)))
        len_img = 32*32*3 + 32*32
        final = combined_img.reshape(1,len_img).astype(np.float32)

        result.append({
            'class': number_2_name(__model.predict(final)[0]),
            'class_probability': np.round(__model.predict_proba(final)*100,2).tolist()[0],
            'class_dictionary': __class_name_to_number
        })
    
    return result
def load_artifacts():
    print("load artifacts start...")

    global __class_name_to_number
    global __class_number_to_name
    with open("./server/artifacts/class_dictionary.json","r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v:k for k,v in __class_name_to_number.items()}
    

    global __model
    if  __model is None:
        with open('./server/artifacts/best_estimator.pkl','rb') as f:
            __model = joblib.load(f)

    print("load artifacts end...")



def get_b64_test_img_for_virat():
    with open('./server/kane_base64.txt') as f:
        return f.read()



def get_cv2_img_from_base46_string(b64str):
    '''
    credit: https://stackoverflow.com/questions/33754935/read-a-base-64-encoded-image-from-memory-using-opencv-python-library
    :param uri:
    :return:
    '''
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def get_face_if_2_eyes(img, base64_data):
    face_cascade = cv2.CascadeClassifier("../../../../HaarCascade/haarcascade_frontalface_default.xml")
    eyes_cascade = cv2.CascadeClassifier("../../../../HaarCascade/haarcascade_eye.xml")

    if img:
        img = cv2.imread(img)
    else:
        img = get_cv2_img_from_base46_string(base64_data)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray ,1.3 , 4)

    cropped_faces = []
    for (x,y,h,w) in faces :
        roi_gray=gray[y:y+h ,x:x+w] # get face part
        roi_color=img[y:y+h ,x:x+w]
        eyes = eyes_cascade.detectMultiScale(roi_gray) # detect eyes
        if len(eyes) >= 2:
            cropped_faces.append(roi_color)
    
    return cropped_faces


if __name__ == "__main__":
    load_artifacts()
    print(classify_image(get_b64_test_img_for_virat()))
    # print(classify_image(None, "./server/test_images/ronaldo_(56).jpg"))
    # print(classify_image(None, "./server/test_images/kane_(117).jpg"))
    # print(classify_image(None, "./server/test_images/Sharapova_ (93).jpg"))