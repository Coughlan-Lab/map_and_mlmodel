"""image_processing.py
This module processes a set of images and associated VIO readings,
preparing selected images for labeling using VoTT, then constructing
an SFrame for training an object detection model through turicreate.
The image set is then used in conjection with a floorplan to determine the
scale of the floorplan map and the location of the signs.
Copyright 2021 Smith-Kettlewell Institute - code by Ryan Crabb
"""

import cv2
import glob
import os
import shutil
import pickle
import tkinter as tk
import turicreate as tc
from os.path import join
import numpy as np


IMAGE_EXTENSION = '.jpg'
MAP_DIR_STR = './maps/'
DATA_DIR_STR = './datasets/'


def popup_msg(msg):
    popup = tk.Tk()
    popup.wm_title("!")
    label = tk.Label(popup, text=msg, font=("Verdana", 12))
    label.pack(side="top", fill="x", pady=10)
    B1 = tk.Button(popup, text="Okay", command=popup.destroy)
    B1.pack()
    popup.mainloop()


def whiten_image(img):
    return (255 - (255 - img) / 2).astype(img.dtype)


def select_training_images(list_of_images, data_path, start_index=1,
                           window_title='Select Images for Training - Enter to select - Esc when done'):
    selected_indices = []
    index = start_index
    img = cv2.rotate(cv2.imread(join(data_path, str(start_index) + IMAGE_EXTENSION)), cv2.ROTATE_90_CLOCKWISE)
    cv2.imshow(window_title, img)
    key_press = cv2.waitKey(0)
    while key_press != 27:
        if key_press == 2: # left
            index -= 1
        elif key_press == 3: # right
            index += 1
        elif key_press == 0: # up
            index -= 20
        elif key_press == 1: # down
            index += 20
        elif key_press == 13:
            if index in selected_indices:
                selected_indices.remove(index)
            else:
                selected_indices.append(index)
        else:
            print("Press a directional key, hit Enter when selected")
        # keep within bounds without wrapping
        index = max(1, min(len(list_of_images)-1,index))
        img =  cv2.rotate(cv2.imread(join(data_path, str(index) + IMAGE_EXTENSION)), cv2.ROTATE_90_CLOCKWISE)
        # grey the image to denote it has been selected
        if index in selected_indices:
            img = whiten_image(img)
        cv2.imshow(window_title, img)
        key_press = cv2.waitKey(0)
    cv2.destroyWindow(window_title)
    _ = cv2.waitKey(1)
    print("Image selected was " + str(index))
    return selected_indices


class CameraLogData:
    def __init__(self, source_data_path, floor, dataset_name=None):
        if not dataset_name:
            dataset_name =  os.path.basename(source_data_path)
        # Create the folder for our data set and copy the data there.
        self.data_path = join(DATA_DIR_STR, dataset_name)
        self.sframe_path = join(self.data_path, dataset_name+'.sframe')
        self.image_path = join(self.data_path,'images')
        self.training_path = join(self.data_path, 'training_data')
        self.extra_training_path = join(self.data_path,'extra_training_images')
        self.VIO_path = join(self.data_path,f'vio_{dataset_name}.txt')
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
        if not os.path.exists(self.data_path+'/images'):
            shutil.copytree(source_data_path, self.data_path + '/images')
            vio_file = glob.glob(join(self.data_path, 'VIO*.txt'))[0]
            shutil.move(vio_file, self.VIO_path)

        self.map_file = MAP_DIR_STR + 'walls_' + str(floor) + '.bmp'
        self.path_to_signs = MAP_DIR_STR + 'Floor_' + str(floor) + '_signs.json'
        self.ARKit_data = [x.strip() for x in open(self.VIO_path, "r")]
        self.list_of_images_data = sorted([int(os.path.basename(file).replace(IMAGE_EXTENSION, ''))
                                           for file in glob.glob(join(self.image_path,'*'+IMAGE_EXTENSION))])
        self.training_list = []
        self.training_image_size = (1440, 1920)

    def prepare_training_images(self):
        training_list = select_training_images(self.list_of_images_data, self.image_path)
        if not os.path.exists(self.training_path):
            os.mkdir(self.training_path)
        for index in training_list:
            img =  cv2.rotate(cv2.imread(join(self.image_path, str(index) + IMAGE_EXTENSION)), cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(join(self.training_path, str(index) + IMAGE_EXTENSION), img)
        self.training_image_size = (img.shape[1], img.shape[0])
        self.training_list = training_list
        if not os.path.exists(self.extra_training_path):
            os.mkdir(self.extra_training_path)
        #popup_msg(f"Add extra training images to {join(self.data_path,'extra_training_images')} before running VoTT")
        print(f"Add extra training images to {join(self.data_path, 'extra_training_images')} before running VoTT")
        print("Press any key to continue")
        cv2.waitKey(0)
        extra_image_list = glob.glob(join(self.extra_training_path,'*.jp*g'))
        for i in range(len(extra_image_list)):
            extra_img = cv2.resize(cv2.imread(extra_image_list[i]), self.training_image_size, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(join(self.training_path,f"extra{i:03d}{IMAGE_EXTENSION}"), extra_img)

    def GenerateSFrameFromVoTT(self):
        LABELED_IMAGES_DIR = join(self.training_path, 'vott-csv-export')
        csv_path = glob.glob(join(LABELED_IMAGES_DIR,'*export.csv'))[0]
        if not os.path.exists(csv_path):
            print(f"VoTT export not found at {LABELED_IMAGES_DIR}.")
            return
        csv_sf = tc.SFrame.read_csv(csv_path)
        print(f"Finished reading CSV file: {csv_path}")

        def row_to_bbox_coordinates(row):
            """
            Takes a row and returns a dictionary representing bounding
            box coordinates:  (center_x, center_y, width, height)  e.g. {'x': 100, 'y': 120, 'width': 80, 'height': 120}
            """
            return {'x': row['xmin'] + (row['xmax'] - row['xmin']) / 2,
                    'width': (row['xmax'] - row['xmin']),
                    'y': row['ymin'] + (row['ymax'] - row['ymin']) / 2,
                    'height': (row['ymax'] - row['ymin'])}

        csv_sf['coordinates'] = csv_sf.apply(row_to_bbox_coordinates)
        # delete no longer needed columns
        del csv_sf['xmin'], csv_sf['xmax'], csv_sf['ymin'], csv_sf['ymax']
        # rename columns
        csv_sf = csv_sf.rename({'image': 'name'})

        # Load all images in random order
        sf_images = tc.image_analysis.load_images(LABELED_IMAGES_DIR, recursive=True, random_order=True)
        print(f"Finished loading image files from path: {LABELED_IMAGES_DIR}")

        # Split path to get filename
        info = sf_images['path'].apply(lambda path: os.path.basename(path).split('/')[:1])
        # Rename columns to 'name'
        info = info.unpack().rename({'X.0': 'name'})

        # Add to our main SFrame
        sf_images = sf_images.add_columns(info)
        # Original path no longer needed
        del sf_images['path']

        # Combine label and coordinates into a bounding box dictionary
        csv_sf = csv_sf.pack_columns(['label', 'coordinates'], new_column_name='bbox', dtype=dict)

        # Combine bounding boxes of the same 'name' into lists
        sf_annotations = csv_sf.groupby('name', {'annotations': tc.aggregate.CONCAT('bbox')})

        # Join annotations with the images. Note, some images do not have annotations,
        # but we still want to keep them in the dataset. This is why it is important to
        # a LEFT join.
        sf = sf_images.join(sf_annotations, on='name', how='left')

        # The LEFT join fills missing matches with None, so we replace these with empty
        # lists instead using fillna.
        sf['annotations'] = sf['annotations'].fillna([])

        # Save SFrame
        print(f"Saving SFrame at {self.sframe_path}")
        sf.save(self.sframe_path)


if __name__ == '__main__':
    dataset = 'Garage'
    save_file = join(DATA_DIR_STR,dataset+'.pik')
    orig_path = '/Users/rcrabb/PycharmProjects/map_and_mlmodel/datasets/Garage/images'
    # if os.path.exists(save_file):
    #     with open(save_file, 'rb') as fp:
    #         cam_log = pickle.load(fp)
    # else:
    cam_log = CameraLogData(orig_path, 1, dataset_name=dataset)
    cam_log.GenerateSFrameFromVoTT()
        # if not cam_log.training_list:
        #     cam_log.prepare_training_images()
    with open(save_file, 'wb') as fp:
        pickle.dump(cam_log, fp)

