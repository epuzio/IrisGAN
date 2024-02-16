import cv2, sys, os, math, glob
from PIL import image

# To fix:
# 1) Output images should be numpy arrays, 192x256? Look at ArtGAN training set for correct format
#   -> This is a change that can come after gan.py is working...
# 2) Should be able to specify input/output files from command line, with default as input_videos and output_training_set
# 3) Center faces in output (DONE)
    # Figure out cropping 
# 4) The code works though :~)

def get_title(file_path): 
    '''
    Helper function to split file path to get title of video
    '''
    return os.path.split(os.path.split(file_path)[1])[1].split('.')[0]

def process_videos():
    '''
    Helper function to run detect_faces on single .mp4 input or on all videos in specified folder
    '''
    
    if os.path.isfile(sys.argv[2]):
        detect_faces(sys.argv[2])
    if os.path.isdir(sys.argv[2]):
        for file in glob.glob(sys.argv[2] + "/*"):
            print("Processing:", file)
            detect_faces(file)
    else:
        print("Invalid file path")

def crop_bounds(x, y, w, h, dim_x, dim_y, frame):
    center_x, center_y = x + (w // 2), y + (h // 2)
    x_start = max(0, center_x - dim_x)
    y_start = max(0, center_y - dim_y)
    x_end = min(frame.shape[1], center_x + dim_x)
    y_end = min(frame.shape[0], center_y + dim_y)
    frame = frame[y_start:y_end, x_start:x_end]
    # center_x, center_y = x + (w // 2), y + (h // 2) #modulo wrap version!
    # frame = frame[(center_x - dim_x) % frame.shape[0]:(center_x + dim_x) % frame.shape[0], (center_y - dim_y) % frame.shape[1]:(center_y + dim_y) % frame.shape[1]]
    return frame

def detect_faces(file_path, crop_frames = True, dim_x = 192, dim_y = 256):
    '''
    Uses cv2 CascadeClassifier to find all frames containing human faces,
    frames are output to local output_np_images folder
    '''
    print("DF fp:", file_path)
    video = cv2.VideoCapture(file_path)
    fps = math.ceil(video.get(cv2.CAP_PROP_FPS))
    dim_x, dim_y = dim_x // 2, dim_y // 2 #3:4 ratio by default
    kps = 1 #number of captures per second of film - 1 is every frame, 2 is every other frame, etc

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
    title = get_title(file_path)
    print("file:", title)
    os.chdir("output_np_images") #specify output directory of images
    for _ in range(int(video.get(cv2.CAP_PROP_FRAME_COUNT))):
        image_filename = f'{title}_frame{fc}.jpg'  
        read_frame_success, frame = video.read()
        
        if fc % round(fps / kps) == 0:
            if not read_frame_success:
                break
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            #Left Profile:
            left_profile = left_profile_classifier.detectMultiScale(
                gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
            )
            for (x, y, w, h) in left_profile:
                if crop_frames:
                    frame = crop_bounds(x, y, w, h, dim_x, dim_y, frame)
                cv2.imwrite(image_filename, frame)
            
            #Right profile
            right_profile = left_profile_classifier.detectMultiScale( #flip image to check right profiles
                cv2.flip(gray_image, 1), scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
            )
            for (x, y, w, h) in right_profile:
                if crop_frames:
                    frame = crop_bounds(x, y, w, h, dim_x, dim_y, frame)
                cv2.imwrite(image_filename, frame)
            
            #Front face    
            front_face = front_face_classifier.detectMultiScale(
                gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
            )
            for (x, y, w, h) in front_face:
                if crop_frames:
                    frame = crop_bounds(x, y, w, h, dim_x, dim_y, frame)
                cv2.imwrite(image_filename, frame)
        fc += 1
        print("Outputting frame:", fc, end="\r")

def remove_files():
    '''
    Removes images from output_np_images
    '''
    if len(sys.argv) == 2:
        for img in os.listdir('output_np_images'):
            os.remove(os.path.join("output_np_images", img))
    if len(sys.argv) == 3:
        title = get_title(sys.argv[2])
        for img in os.listdir('output_np_images'):
            if title == img.split("_")[0]:
                os.remove(os.path.join("output_np_images", img))

def main(): 
    if len(sys.argv) < 1:
        print("ARG1: input video path as .mp4")
    else:
        match sys.argv[1].split('.')[-1]:
            case "load":
                process_videos()
            case "clean":
                remove_files()
            case "help":
                print("To load single video: python3 detect.py load input_videos/title_of_video.mp4")
                print("To load multiple videos from file: python3 detect.py load input_videos")
                print("To clean all files: python3 detect.py clean")
                print("To clean files by title: python3 detect.py clean title")

if __name__ == "__main__":
    main()
