import os
import cv2
from PIL import Image, ImageTk
from sys import path
import numpy as np
cpath = os.getcwd()
# path.append(r'E:\side_job\Korean_Consulting_project\Face-Morphing-master\codeX\utils')
path.append(cpath +r'\codeX\utils')

from face_landmark_detection import generate_face_correspondences
from delaunay_triangulation import make_delaunay
from face_morph import generate_morph_sequence
import subprocess
import argparse
import shutil
import os
from sys import path
import time
import analysis_morph 
# from deepface import DeepFace

from flask import (
    Blueprint,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)

dt = Blueprint("morphing", __name__, template_folder="templates")


def doMorphing(self, img1, img2, duration, frame_rate, output):
    
        [size, img1, img2, points1, points2, list3] = generate_face_correspondences(img1, img2)
        tri = make_delaunay(size[1], size[0], list3, img1, img2)
        generate_morph_sequence(duration, frame_rate, img1, img2, points1, points2, tri, size, output)

def start(self):
        global RAW_IMAGES_DIR
        global ALIGNED_IMAGES_DIR
        global img_name
        global image1
        global image2

        img_name = os.listdir(ALIGNED_IMAGES_DIR)       
        image = Image.open(ALIGNED_IMAGES_DIR+'/'+img_name[0])
        # resize image to match canvas size
        image = image.resize((500, 500))
        # convert image to tkinter PhotoImage and display on canvas
        self.image = ImageTk.PhotoImage(image)
        self.image_canvas.create_image(0, 0, anchor=tk.NW, image=self.image)
        ########################################################################

        img1 = ALIGNED_IMAGES_DIR+'/'+img_name[0]
        img2  = ALIGNED_IMAGES_DIR+'/'+img_name[1]
        print(img1,img2)
        image1 = cv2.imread(img1)
        image2 = cv2.imread(img2)
        out_folder = cpath+r'\video_output.mp4'
        print(out_folder)
        self.doMorphing(image1,image2,int(5),int(20),out_folder) ## Video Time
        # filename = 'E:\side_job\Korean_Consulting_project\Face-Morphing-master\output.mp4'
        filename = out_folder

        self.cap = cv2.VideoCapture(filename)

        # get video dimensions
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # calculate scale factor to match canvas size
        if width > height:
            scale = 500 / width
        else:
            scale = 500 / height

        # update video canvas with each frame of the video
        ret, frame = self.cap.read()
        if ret:
            # resize frame to match canvas size
            frame = cv2.resize(frame, (int(scale*width), int(scale*height)))
            self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frame = Image.fromarray(np.uint8(self.frame))
            self.frame = ImageTk.PhotoImage(self.frame)
            self.video_canvas.create_image(0, 0, anchor=tk.NW, image=self.frame)
            self.master.after(10, self.update_video)

def play_video(self):
        # filename = 'E:\side_job\Korean_Consulting_project\Face-Morphing-master\output.mp4'
        filename = cpath+r'\video_output.mp4'

        self.cap = cv2.VideoCapture(filename)

        # get video dimensions
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # calculate scale factor to match canvas size
        if width > height:
            scale = 500 / width
        else:
            scale = 500 / height

        # update video canvas with each frame of the video
        ret, frame = self.cap.read()
        if ret:
            # resize frame to match canvas size
            frame = cv2.resize(frame, (int(scale*width), int(scale*height)))
            self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # res = Image.fromarray(cv2.cvtColor(np.uint8(morphed_frame), cv2.COLOR_BGR2RGB))
            # self.frame = Image.fromarray(self.frame)
            self.frame = Image.fromarray(np.uint8(self.frame))
            self.frame = ImageTk.PhotoImage(self.frame)
            self.video_canvas.create_image(0, 0, anchor=tk.NW, image=self.frame)
            self.master.after(10, self.update_video)

def update_video(self):
        # update video canvas with each frame of the video
        ret, frame = self.cap.read()
        if ret:
                # calculate scale factor to match canvas size
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if width > height:
                        scale = 500 / width
                else:
                        scale = 500 / height
                
                # resize frame to match canvas size
                frame = cv2.resize(frame, (int(scale*width), int(scale*height)))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frame = Image.fromarray(frame)
                self.frame = ImageTk.PhotoImage(self.frame)
                self.video_canvas.create_image(0, 0, anchor=tk.NW, image=self.frame)
                self.master.after(10, self.update_video)            