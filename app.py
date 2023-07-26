from flask import Flask, request, jsonify
import cv2
import numpy as np
import face_recognition
import os

from werkzeug.utils import secure_filename

from eyeblink_detector import *
Upload = 'UPLOAD_FOLDER'
app = Flask(__name__)
app.config['uploadFolder'] = Upload


def delete_all_images(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"File '{file_path}' has been deleted.")
# Function to process the image and name
def findEncodings(images):
    # print("in find encodings",images)
    count = 0
    encodeList = []
    # print("len of images ",len(images))
    # print(images[-1])
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        # print(encode)
        encodeList.append(encode)
        count += 1
        # print("count - ",count)
    # print(encodeList)
    # os.remove(os.path.join('ImagesAttendance',images[0]))
    return encodeList


def markAttendance(name):
    print("face detected of ->", name)
    return True
def read_filestorage_image(file_storage_obj):
    # Step 1: Get the byte string data from FileStorage object
    file_data = file_storage_obj.read()

    # Step 2: Convert the byte string to a NumPy array
    nparr = np.frombuffer(file_data, np.uint8)

    # Step 3: Read the NumPy array using OpenCV
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    return img
def convert_filestorage_to_numpy(file_storage):
    try:
        img_np = np.fromfile(file_storage, np.uint8)
        img_np = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        return img_np
    except Exception as e:
        print(f"Error converting FileStorage to NumPy array: {e}")
        return None
def matching_face_invideo(video_file):
    print("in matching face ",video_file)
    ## var initiation
    path = 'ImagesAttendance'
    images = []
    classNames = []
    myList = os.listdir(path)
    print(myList)
    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    # classNames['biplab', 'dibyajit']
    print(classNames)

    # print("get the video file",video_file)
    # print("check the type of video - ",type(video_file))
    ## added function to work with cv2
    encodeListKnown = findEncodings(images)
    print('Encoding Complete')
    delete_all_images(path)


    cap = cv2.VideoCapture(video_file)

    while True:
        try:
            success, img = cap.read()
            # img = captureScreen()
            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            facesCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                # print(faceDis)
                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    name = classNames[matchIndex].upper()
                    # print(name)
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    if markAttendance(name):
                        print("verification completed!!!!")
                        return "verification completed!!!!",name
                else:
                    print("matching working")
                    return "verification failed!!!!","None"
                        # cap.release()
                        # cv2.destroyAllWindows()
                    # break
                    # break

            # cv2.imshow('Webcam', img)
            # cv2.waitKey(1)
        except Exception as e:
            print("error in encoding ",e)
            break

    # return "username"
def process_image_and_name(image_file, name):
    # Implement your conditions here to process the image and name
    # For simplicity, we'll assume the condition is True if the name is "John"
    # and the image processing is successful.
    print(image_file)
    print(type(image_file))
    # Load the cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Read the input image
    img = cv2.imread(image_file)
    # img = read_filestorage_image(image_file)

    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cropped_face = img[y:y + h, x:x + w]
        cv2.imwrite(f'ImagesAttendance/{name}.jpg', cropped_face)
        print("image writing done")
        return True
        # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return False

# API endpoint to process the image and name
@app.route('/image_process', methods=['POST'])
def image_process():
    # name = request.form['name']
    # print("name - here ",name)
    # image_file = request.files['image']
    #
    # # Process the image and name
    # result = process_image_and_name(image_file, name)
    #
    # # Return the result as JSON
    # return jsonify({'result': result})
    try:
        # Get the name and image file from the request
        name = request.form['name']
        print("name ",name)
        image_file = request.files['image']
        print("image file ",image_file.filename)
        image_file.save(image_file.filename)

        # Process the image and name
        result = process_image_and_name(image_file.filename, name)

        # Return the result as JSON
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)})
# API endpoint to process the video and check the condition
@app.route('/process_video', methods=['POST'])
def process_video_api():
    try:
        # Get the video file from the request
        video_file = request.files['video']
        print("check the video file ",video_file.filename)
        # print("checking blinking result ",check_blinking(video_file.filename))
        # # Create a destination path to store the file in the current directory
        # destination_path = os.path.join(os.getcwd(), video_file)
        # ## saving the video in working directory
        # video_file.save(destination_path)
        # if video_file.filename != "":
        filename = secure_filename(video_file.filename)
        video_path = os.path.join(app.config['uploadFolder'],filename)
        video_file.save(video_path)
        print("file saving done")
        print(video_path)
        # print(check_blinking(video_path))

        if check_blinking(video_path):
            print("start for face checking")


            # Process the video and check the condition
            result,user_name = matching_face_invideo(video_path)

            ## delete the video before return the result
            delete_all_images('UPLOAD_FOLDER')

            # Return the result as JSON
            return jsonify({'result': result,'user_name':user_name})
        else:
            # Return the result as JSON
            return jsonify({'result': "No Blinking Detected", 'user_name': "None"})
    except Exception as e:
        return jsonify({'result': "verification failed!!!!",'user_name':"None"})

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
