import cv2, sys, os, math, glob, time, zipfile


# To fix:
# 1) Output images should be numpy arrays, 192x256? Look at ArtGAN training set for correct format
#   # This is a change that can come after gan.py is working... (DONE - TFRecord Binary, not numpy array)
# 2) Should be able to specify input/output files from command line, with default as input_videos and output_training_set
    # or as global variables
# 3) Center faces in output (DONE)
    # Figure out cropping (DONE)
    # Downsize output images by 2x (DONE)
    # Add user control for cropping dimensions from the command line
# 4) It says invalid file path at the end
# 5) Generally make the code cleaner and more readable
# 6) TFRecord support >:) (DONE?)
    # Tensorflow takes ages to import - might be worth moving make_training_set to a separate file
# 7) Make some global variables for names of files and directories
# 8 Put the classifier detection in a for loop

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

def crop_bounds(x, y, w, h, dim_x, dim_y, frame): #why did this algorithm take me so long.... 
    '''
    Helper function to crop frames around faces to dimensions specified by dim_x, dim_y.
    Extra logic added so that the crop stays within the bounds of the original image.
    Dim_x and Dim_y are the x and y dimensions of the cropped frame
    '''
    center_x =  x + (w//2)
    center_y =  y + (h//2)
    top_x = center_x - (dim_x // 2)
    top_y = center_y - (dim_y // 2)
    bottom_x = center_x + (dim_x // 2)
    bottom_y = center_y + (dim_y // 2)
    
    if top_x < 0:
        bottom_x -= top_x
        top_x = 0
    if top_y < 0:
        bottom_y -= top_y
        top_y = 0
    if bottom_x > frame.shape[1]:
        top_x -= (bottom_x - frame.shape[1])
        bottom_x = frame.shape[1]
    if bottom_y > frame.shape[0]:
        top_y -= (bottom_y - frame.shape[0])
        bottom_y = frame.shape[0]
        
    cropped_frame = frame[top_y:bottom_y, top_x:bottom_x]
    return cropped_frame
        
# def detect_faces(file_path, crop_frames = True, dim_x = 288, dim_y = 384):
def detect_faces(file_path, crop_frames = True, dim_x = 256, dim_y = 256):
    '''
    Uses cv2 CascadeClassifier to find all frames containing human faces,
    Frames are output to local output_images folder. All images must be the same dimensions
    for the GAN to work properly.
    '''

    
    
    video = cv2.VideoCapture(file_path)
    fps = math.ceil(video.get(cv2.CAP_PROP_FPS))
    kps = 1 #number of captures per second of film - 1 is every frame, 2 is every other frame, etc

    if not video.isOpened():
        print("Error opening video file")
        exit()
        
    print("Video Dimensions:", video.get(cv2.CAP_PROP_FRAME_WIDTH), "x" , video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if(dim_x > video.get(cv2.CAP_PROP_FRAME_WIDTH) or dim_y > video.get(cv2.CAP_PROP_FRAME_HEIGHT)): #:pensive:
        print("Error: cropped dimensions exceed the dimensions of the original video.")
        exit()
    

    front_face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    left_profile_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_profileface.xml"
    )

    fc = 0
    title = get_title(file_path)
    print("File:", title)
    print("Frames:", int(video.get(cv2.CAP_PROP_FRAME_COUNT)))
    start_time = time.time()
    os.chdir("output_images") #specify output directory of images
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
            if len(left_profile) > 0:
                for (x, y, w, h) in left_profile:
                    if crop_frames:
                        frame = crop_bounds(x, y, w, h, dim_x, dim_y, frame)
                        # frame = cv2.resize(frame, (192, 256)) #resize???
                    cv2.imwrite(image_filename, frame)
            
            #Right profile
            right_profile = left_profile_classifier.detectMultiScale( #flip image to check right profiles
                cv2.flip(gray_image, 1), scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
            )
            if len(right_profile) > 0:
                for (x, y, w, h) in right_profile:
                    if crop_frames:
                        frame = crop_bounds(x, y, w, h, dim_x, dim_y, frame)
                        # frame = cv2.resize(frame, (192, 256)) #resize by x2
                    cv2.imwrite(image_filename, frame)
            
            #Front face    
            front_face = front_face_classifier.detectMultiScale(
                gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
            )
            if len(front_face) > 0:
                for (x, y, w, h) in front_face:
                    if crop_frames:
                        frame = crop_bounds(x, y, w, h, dim_x, dim_y, frame)
                        # frame = cv2.resize(frame, (192, 256)) #resize by x2
                    cv2.imwrite(image_filename, frame)
        fc += 1
        print("Outputting Frame:", fc, end="\r")
    print("Finished in:", round((time.time() - start_time), 2), "seconds.")

def remove_files():
    '''
    Removes images from output_images
    '''
    if len(sys.argv) == 2:
        for img in os.listdir('output_images'):
            os.remove(os.path.join("output_images", img))
    if len(sys.argv) == 3:
        title = get_title(sys.argv[2])
        for img in os.listdir('output_images'):
            if title == img.split("_")[0]:
                os.remove(os.path.join("output_images", img))


def zip_files():
    '''
    Zips all images in output_images to a .zip file
    '''
    with zipfile.ZipFile("output_training_set.zip", "w") as zipf: #name shouldn't be hardcoded, fix later
        for img in os.listdir("output_images"):
            zipf.write(os.path.join("output_images", img), img)
            
            
def make_training_set():
    '''
    Save all images in output_images to a .tfrecord file - this is an efficient way 
    to store a dataset for the GAN. Tensorflow is imported here as it takes a long time to import
    and should only be imported when necessary.
    '''
    from tensorflow.io import TFRecordWriter, encode_jpeg #might be worth it to move make_training_set + zip to a separate file
    from tensorflow.train import Example, Features, Feature, BytesList
    
    
    tfrecord_file_path = 'output.tfrecord'
    
    if(os.path.exists("output_training_set.zip")): #There's a .zip file for the training set called "output_training_set.zip"
        count = 0
        with TFRecordWriter(tfrecord_file_path) as writer:
            with zipfile.ZipFile("output_training_set.zip", "r") as zip_ref:
                # Iterate over the files in the zip file
                for file_name in zip_ref.namelist():
                    # Read the image from the zip file
                    image_data = zip_ref.read(file_name)
                    
                    feature = {
                        'image': Feature(bytes_list=BytesList(value=[image_data]))
                    }
                    
                    example = Example(features=Features(feature=feature))
                    writer.write(example.SerializeToString()) #SerializeToString turns the example into a binary string
                    count += 1
        print(f"TFRecord file created from zip: {tfrecord_file_path}, {count} images written.")
        
    else: #There's no .zip file for the training set, make a .tfrecord file from output_images
        #Doesn't work right now.... awesome
        with TFRecordWriter(tfrecord_file_path) as writer:
            count = 0
            for img in os.listdir("output_images"):
                feature = { #make 
                    'image': Feature(bytes_list=BytesList(value=[encode_jpeg(img).numpy()]))
                    # 'image': Feature(bytes_list=BytesList(value=[encode_jpeg(img).numpy()])) #as bytes list
                }
                example = Example(features=Features(feature=feature))
                writer.write(example.SerializeToString()) #turn to byteslist!
                count += 1
        print(f"TFRecord file created from folder: {tfrecord_file_path}, {count} images written.")
        

def main(): 
    if len(sys.argv) < 1:
        print("ARG1: input video path as .mp4")
    else:
        match sys.argv[1].split('.')[-1]:
            case "load":
                process_videos()
            case "clean":
                remove_files()
            case "zip":
                zip_files()
            case "tfds":
                make_training_set()
            case "help":
                print("To load single video: python3 detect.py load input_videos/title_of_video.mp4")
                print("To load multiple videos from file: python3 detect.py load input_videos")
                print("To clean all files: python3 detect.py clean")
                print("To clean files by title: python3 detect.py clean title")
                print("Make zip file from image set: python3 detect.py zip")
                print("Make tfds (for GAN): python3 detect.py tfds") 

if __name__ == "__main__":
    main()
