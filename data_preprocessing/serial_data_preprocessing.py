import cv2
import glob
import time
import numpy as np
import math
from PIL import Image
from os import listdir, makedirs
from os.path import join, exists
from facenet_pytorch import MTCNN
from random import shuffle

from skimage.io import imsave
import imageio.core.util

training_videos_folder = ["/media/diode/WorkSpace/DataSets/Face Forensic/train/0",
                         "/media/diode/WorkSpace/DataSets/Face Forensic/train/1"]

frameRate = 25  # frame rate
def frame_extraction():
    print("start extracting frames...")
    for folder in training_videos_folder:
        videos_path = glob.glob(join(folder, "*.mp4"))
        folder = folder.split("/")[-1]
        print(f"num of videos in {folder}: {len(videos_path)}")
        counter = 0
        for video_path in videos_path:
            cap = cv2.VideoCapture(video_path)
            
            #获取视频总帧数
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            #视频分6段，每段取前5帧连续图像
            frame_split_index = math.floor(frame_count/6)
            frame_count_index = []  #需要截取帧数的list
            for i in range(6):
                for j in range(5):
                    frame_count_index.append(i*frame_split_index + j)
            
            
            vid = video_path.split("/")[-1]
            vid = vid.split(".")[0] 

            if not exists("../train_frames/" + folder + "/video_" + str(int(counter))):
                makedirs("../train_frames/" + folder + "/video_" + str(int(counter)))

            i = 0
            while cap.isOpened():
                frameId = cap.get(1)  # current frame number
                ret, frame = cap.read()
                if not ret:
                    break
                filename = (
                    "../train_frames/"
                    + folder
                    + "/video_"
                    + str(int(counter))
                    + "/image_"
                    + str(int(frameId) + 1)
                    + ".jpg"
                )
                if i in frame_count_index == 0:
                    cv2.imwrite(filename, frame)  
                i += 1
            cap.release()
            if counter % 100 == 0:
                print("Number of videos done:", counter)
            counter += 1

def ignore_warnings(*args, **kwargs):
    pass

def face_extraction():
    print("start extracting faces...")
    imageio.core.util._precision_warn = ignore_warnings
    # Create face detector
    # If you want to change the default size of image saved from 160, you can
    # uncomment the second line and set the parameter accordingly.
    mtcnn = MTCNN(
        margin=40,
        select_largest=False,
        post_process=False,
        device="cuda:0",
        image_size=224
    )
    # mtcnn = MTCNN(margin=40, select_largest=False, post_process=False,
    # device='cuda:0', image_size=256)

    for ind in range(2):
        # Directory containing images respective to each video
        source_frames_folders = [f"../train_frames/{ind}/"]
        # Destination location where faces cropped out from images will be saved
        dest_faces_folder = f"../train_face/{ind}/"
        for i in source_frames_folders:
            counter = 0
            for j in listdir(i):
                imgs = glob.glob(join(i, j, "*.jpg"))
                if counter % 1000 == 0:
                    print("Number of videos done:", counter)
                if not exists(join(dest_faces_folder, j)):
                    makedirs(join(dest_faces_folder, j))
                for k in imgs:
                    frame = cv2.imread(k)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = Image.fromarray(frame)
                    face = mtcnn(frame)
                    try:
                        imsave(
                            join(dest_faces_folder, j, k.split("/")[-1]),
                            face.permute(1, 2, 0).int().numpy(),
                        )
                    except AttributeError:
                        print("Image skipping")
                counter += 1

def convert_data_to_npy():
    print("start create npy files...")
    img_size = 224
    frames_per_video = 30
    train_path = ["../train_face/1", "../train_face/0"]
    list_1 = [join(train_path[0], x) for x in listdir(train_path[0])]
    list_0 = [join(train_path[1], x) for x in listdir(train_path[1])]
    
    for i in range(len(list_0) // len(list_1)):
        vid_list = list_1 + list_0[i * (len(list_1)): (i + 1) * (len(list_1))]
        shuffle(vid_list)
        train_data = []
        train_label = []
        count = 0
        images = []
        labels = []
        counter = 0
        for x in vid_list:
            img = glob.glob(join(x, "*.jpg"))
            img.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
            images += img[: frames_per_video]
            label = [k.split("/")[1] for k in img]
            labels += label[: frames_per_video]
            if counter % 1000 == 0:
                print("Number of files done:", counter)
            counter += 1
        for j, k in zip(images, labels):
            img = cv2.imread(j)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(
                img, (img_size, img_size), interpolation=cv2.INTER_AREA
            )
            train_data.append(img)
            train_label += [k]
            if count % 10000 == 0:
                print("Number of files done:", count)
            count += 1
            if count >= 100000:
                continue
        train_data = np.array(train_data)
        train_label = np.array(train_label)
        print(train_data.shape, train_label.shape)
        np.save("train_data_" + str(frames_per_video) + f"_{img_size}.npy", train_data)
        np.save("train_label_" + str(frames_per_video) + f"_{img_size}.npy", train_label)
        print(f"{(i+1) / len(list_0) // len(list_1)}")

# t1 = time.time()
# frame_extraction()
# t2 = time.time()
# print(f"spending {int(t2-t1)//3600} hours for extracting frames")

face_extraction()
t3 = time.time()
print(f"spending {int(t3-t2)//3600} hours for extracting faces")

convert_data_to_npy()
t4 = time.time()
print(f"spending {int(t4-t3)//3600} hours for convert data to npy")

print("finished!")
