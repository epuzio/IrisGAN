# following this tutorial: https://www.datacamp.com/tutorial/face-detection-python-opencv
#images from file: https://palashsharma891.medium.com/imagesort-using-python-496470ea2102
#goal of project - feed in a video (frames), choose frames that have a human face/are distinct from previous frame

#cv2 has video capture https://docs.opencv.org/3.4/dd/d43/tutorial_py_video_display.html
#haar cascades: https://github.com/opencv/opencv/tree/master/data/haarcascades

import cv2, sys, os

if argv.len() < 1:
    print("ARG1: input video path as .mp4, ARG2: output destination")
else:
    video = cv2.VideoCapture(sys.argv[1])
    fps = video.get(cv2.CAP_PROP_FPS)
    output_path = sys.argv[2]

if not video.isOpened():
    print("Error opening video file")
    exit()

front_face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

left_profile_classifier = cv2.CascadeClassifier( #image = cv2.flip(image, 1) 
    cv2.data.haarcascades + "haarcascade_profileface.xml"
)

fc = 0
while True: #seems redundant but thanks gpt
    read_frame_success, frame = video.read()
    if not read_frame_success:
        break
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = []
    faces.append(front_face_classifier.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    ))
    faces.append(left_profile_classifier.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    ))
    faces.append(left_profile_classifier.detectMultiScale( #flip image to check right profiles
        cv2.flip(gray_image, 1), scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    ))

    if faces: #faces detected in image (list isn't empty)
        image_filename = f'frame_{fc}.jpg'  
        cv2.imwrite(image_filename, frame)
    fc += 1
















# photos = []
# for photo in os.walk(photo_folder_path):
#     photos += [photo]
# for photo in photos:
#     curr_image = cv2.imread(photo)
#     grayscale_curr_image = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY)

#use haarcascade_profileface.xml and haarcascade_frontalface_default.xml profile face only looks left tho

