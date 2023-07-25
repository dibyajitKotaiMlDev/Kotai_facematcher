# All the imports go here
# import numpy as np
import cv2
## blink checked function
def blinkchecker(input_list):
    has_one = False
    has_zero = False

    for num in input_list:
        if num == 1:
            has_one = True
        elif num == 0:
            has_zero = True

        if has_one and has_zero:
            return True

    return False
## this following function returns true if blinking detected if false blinking not detected
def check_blinking(video_file):
    print("in blinking")
    # Initializing the face and eye cascade classifiers from xml files
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

    # Variable store execution state
    first_read = True
    blink_list = []
    # Starting the video capture
    cap = cv2.VideoCapture(video_file)
    ret, img = cap.read()

    while (ret):
        try:
            ret, img = cap.read()
            # Converting the recorded image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Applying filter to remove impurities
            gray = cv2.bilateralFilter(gray, 5, 1, 1)

            # Detecting the face for region of image to be fed to eye classifier
            faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(200, 200))
            if (len(faces) > 0):
                for (x, y, w, h) in faces:
                    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # roi_face is face which is input to eye classifier
                    roi_face = gray[y:y + h, x:x + w]
                    roi_face_clr = img[y:y + h, x:x + w]
                    eyes = eye_cascade.detectMultiScale(roi_face, 1.3, 5, minSize=(50, 50))

                    # Examining the length of eyes object for eyes
                    if (len(eyes) >= 2):
                        # Check if program is running for detection
                        if (first_read):
                            print("Eye detected press s to begin")
                            blink_list.append(1)
                            # cv2.putText(img,
                            #             "Eye detected press s to begin",
                            #             (70, 70),
                            #             cv2.FONT_HERSHEY_PLAIN, 3,
                            #             (0, 255, 0), 2)
                        else:
                            print("Eyes open!")
                            blink_list.append(1)
                            # cv2.putText(img,
                            #             "Eyes open!", (70, 70),
                            #             cv2.FONT_HERSHEY_PLAIN, 2,
                            #             (255, 255, 255), 2)
                    else:
                        if (first_read):
                            print("No eyes detected")
                            blink_list.append(0)
                            # To ensure if the eyes are present before starting
                            # cv2.putText(img,
                            #             "No eyes detected", (70, 70),
                            #             cv2.FONT_HERSHEY_PLAIN, 3,
                            #             (0, 0, 255), 2)
                        else:
                            # This will print on console and restart the algorithm
                            print("Blink detected--------------")
                            blink_list.append(0)
                            # cv2.waitKey(3000)
                            first_read = True

            else:
                print("no face detected")
                # cv2.putText(img,
                #             "No face detected", (100, 100),
                #             cv2.FONT_HERSHEY_PLAIN, 3,
                #             (0, 255, 0), 2)

            # # Controlling the algorithm with keys
            # cv2.imshow('img', img)
            # a = cv2.waitKey(1)
            # if (a == ord('q')):
            #     break
            # elif (a == ord('s') and first_read):
            #     # This will start the detection
            #     first_read = False

        # cap.release()
        # cv2.destroyAllWindows()
        except:
            if blinkchecker(blink_list):
                return True
            # break
