# following this tutorial: https://www.datacamp.com/tutorial/face-detection-python-opencv
#images from file: https://palashsharma891.medium.com/imagesort-using-python-496470ea2102
#goal of project - feed in a video (frames), choose frames that have a human face/are distinct from previous frame

#cv2 has video capture https://docs.opencv.org/3.4/dd/d43/tutorial_py_video_display.html
#haar cascades: https://github.com/opencv/opencv/tree/master/data/haarcascades

#python3 detect.py load input_videos/test2.mp4

#Format:
#load: load in video
#clean: remove all files in output_np_images
#clean file_name: remove all files with file_name in output_np_images

import cv2, sys, os, math, glob

def get_title(file_path):
    return os.path.split(os.path.split(file_path)[1])[1].split('.')[0]

def detect_faces():
    video = cv2.VideoCapture(sys.argv[2])
    fps = math.ceil(video.get(cv2.CAP_PROP_FPS))
    print("fps:", fps) 
    kps = 1 #number of captures per video

    if not video.isOpened():
        print("Error opening video file")
        exit()

    front_face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    left_profile_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_profileface.xml"
    )

    fc = 0
    title = get_title(sys.argv[2])
    print("file:", title)
    os.chdir("output_np_images") #specify output of images
    while True: #seems redundant but thanks gpt
        read_frame_success, frame = video.read()
        if fc % round(fps / kps) == 0:
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
                image_filename = f'{title}_frame{fc}.jpg'  
                cv2.imwrite(image_filename, frame)
        fc += 1
        print("Outputting frame:", fc, end="\r")


def remove_files():
    if len(sys.argv) == 2:
        for img in os.listdir('output_np_images'):
            os.remove(os.path.join("output_np_images", img))
    if len(sys.argv) == 3:
        title = get_title(sys.argv[2])
        for img in os.listdir('output_np_images'):
            if title == img.split("_")[0]:
                os.remove(os.path.join("output_np_images", img))
        # os.remove(img) for img in os.listdir('/output_np_images') if sys.argv[3] in file

if len(sys.argv) < 1:
    print("ARG1: input video path as .mp4")
else:
    match sys.argv[1].split('.')[-1]:
        case "load":
            detect_faces()
        case "clean":
            remove_files()
        case "help":
            print("To load single video: python3 detect.py load input_videos/title_of_video.mp4")
            print("To clean all files: python3 detect.py clean")
            print("To clean files by title: python3 detect.py clean title")