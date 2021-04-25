import numpy as np

import os

import math
from pcdet.utils import common_utils, box_utils, calibration_kitti

import io


base_directory = "/home/dan/Documents/OpenPCDet/data/kitti/training"
original_label_directory = "/label_2"
transformed_label_directory = "/label_2_transformed"
calib_directory = "/home/dan/Documents/OpenPCDet/data/kitti/training/calib"



def display(var_name , var):
    print("\n" + var_name + "\n{}".format(var))


global counterdd 
counterdd = 0

adresses = dict(
    {original_label_directory: transformed_label_directory
    }
)

def execute():
    counterdd = 0
    for idx in range(0, 7480, 1):
        formatted_idx = format(idx, '06d')
        with io.open("./training/label_2/" + formatted_idx +  ".txt", mode="r", encoding="utf-8") as f:
            for line in f:
                label = line.strip().split(' ')
                calib = calibration_kitti.Calibration("./training/calib/" + formatted_idx +  ".txt")
                # h = float(label[8])
                # w = float(label[9])
                # l = float(label[10])

                loc = np.array((float(label[11]), float(label[12]), float(label[13])), dtype=np.float32)
                loc.reshape(1, 3)
                rz = float(label[14]) # named ry of camera, but its actually rz of lidar

                loc_lidar = calib.rect_to_lidar_daniel(loc)

                loc_lidar[2] = loc_lidar[2] - 1.55

                transformed_loc = calib.lidar_to_rect_daniel(loc_lidar)

                str_transformed_x = "{:.2f}".format(transformed_loc[0])
                str_transformed_y = "{:.2f}".format(transformed_loc[1])
                str_transformed_z = "{:.2f}".format(transformed_loc[2])

                with open("./training/label_2_transformed/" + formatted_idx +  ".txt", 'a+') as out_file:
                    out_file.write( str(label[0]) + " " + str(label[1]) + " " + str(label[2]) + " " + str(label[3]) + " " + str(label[4]) + " " + str(label[5]) + " " + str(label[6]) + \
                    " " + str(label[7]) + " " + str(label[8]) + " " + str(label[9]) + " " + str(label[10]) + " " + str_transformed_x + " " +  str_transformed_y + " " + str_transformed_z + " " + str(label[14]) + "\n")

        counterdd = counterdd + 1
        print(counterdd)

execute()
