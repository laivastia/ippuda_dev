import os
import sys
import bz2
import argparse
from face_alignment_dev import image_align
from landmarks_detector import LandmarksDetector
import multiprocessing

def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path

def align_images_dev(RAW_IMAGES):
    if __name__ == "__main__":
        """
        Extracts and aligns all faces from images using DLib and a function from original FFHQ dataset preparation step
        python align_images.py /raw_images /aligned_images
        """
        parser = argparse.ArgumentParser(description='Align faces from input images', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # parser.add_argument('raw_dir', default = r'code\utils\images', help='Directory with raw images for face alignment')
        # parser.add_argument('aligned_dir', default = r'code\utils\images\aligned_images',help='Directory for storing aligned images')
        parser.add_argument('--output_size', default=512, help='The dimension of images for input to the model', type=int)
        parser.add_argument('--x_scale', default=1, help='Scaling factor for x dimension', type=float)
        parser.add_argument('--y_scale', default=1, help='Scaling factor for y dimension', type=float)
        parser.add_argument('--em_scale', default=0.1, help='Scaling factor for eye-mouth distance', type=float)
        parser.add_argument('--use_alpha', default=False, help='Add an alpha channel for masking', type=bool)

        args, other_args = parser.parse_known_args()

        # 내이미지와 타겟 이미지 두개 numpy.arr = RAW_IMAGES
        ALIGNED_IMAGES_DIR = r'images\aligned_images' #args.aligned_dir

        landmarks_detector = LandmarksDetector()
        for img in os.listdir(RAW_IMAGES):
            print('Aligning ...')
            try:
                print('Getting landmarks...')
                for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(img), start=1):
                    try:
                        print('Starting face alignment...')
                        align_img = image_align(img, face_landmarks, output_size=args.output_size, x_scale=args.x_scale, y_scale=args.y_scale, em_scale=args.em_scale, alpha=args.use_alpha)
                        return align_img
                    except:
                        print("Exception in face alignment!")
            except:
                print("Exception in landmark detection!")