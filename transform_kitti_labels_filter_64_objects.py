import numpy as np

import os

import math
from pcdet.utils import common_utils, box_utils, calibration_kitti

import io

global calib_64
calib_64 = dict(
{1.9367: 29, 
1.573966: 28, 
1.304757: 25, 
0.871566: 24, 
0.57880998: 3, 
0.180617: 2, 
-0.088762: 31, 
-0.45182899: 30, 
-0.80315: 27, 
-1.201239: 26, 
-1.49388: 21, 
-1.833245: 20, 
-2.207566: 17, 
-2.546633: 16, 
-2.8738379: 13, 
-3.235882: 12, 
-3.5393341: 23, 
-3.935853: 22, 
-4.2155242: 19, 
-4.5881028: 18, 
-4.9137921: 15, 
-5.2507782: 14, 
-5.6106009: 9, 
-5.958395: 8, 
-6.3288889: 5, 
-6.675746: 4, 
-6.9990368: 1, 
-7.287312: 0, 
-7.6787701: 11, 
-8.0580254: 10, 
-8.3104696: 7, 
-8.7114143: 6, 
-9.0260181: 61, 
-9.5735149: 60, 
-10.06249: 57, 
-10.470731: 56, 
-10.956945: 35, 
-11.598996: 34, 
-12.115005: 63, 
-12.562096: 62, 
-13.040989: 59, 
-13.484814: 58, 
-14.048302: 53, 
-14.598064: 52, 
-15.188733: 49, 
-15.656734: 48, 
-16.176632: 45, 
-16.55401: 44, 
-17.186821: 55, 
-17.73037: 54, 
-18.323431: 51, 
-18.797075: 50, 
-19.320236: 47, 
-19.736372: 46, 
-20.222572: 41, 
-20.787691: 40, 
-21.318125: 37, 
-21.935509: 36, 
-22.437605: 33, 
-22.856577: 32, 
-23.322384: 43,
-23.971016: 42, 
-24.506605: 39, 
-24.999201: 38})

global dic_64_32
dic_64_32 = dict({
    29 : 22,
    28 : 17,
    25 : 13,
    24 : 18,
    3 : 14,
    2 : 9,
    31 : 5,
    30 : 10,
    27 : 6,
    26 : 1,
    21 : 31,
    20 : 2,
    18 : 28,
    16 : 27,
    13 : 23,
    12 : 24,
    23 : 20,
    22 : 19,
    19 : 15,
    18 : 16,
    14 : 12,
    8 : 11,
    0 : 8,
    6 : 7,
    34 : 4,
    48 : 3,
    38 : 0
})


def get_ring_information(loc, l, w, h, rz, type_):

    if type_ == "DontCare":
        return []
    cx = loc[0]
    cy = loc[1]
    cz = loc[2] #- z_correction



    theta = np.radians(rz)
    c, s = np.cos(theta), np.sin(theta)
    Rz = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])

    #display("Rz", Rz)
    

    vertices_add = np.array([
    [ - w/2.0 , - l/2.0 ,  - h/2.0 ],
    [ - w/2.0 , - l/2.0 ,    h/2.0 ],
    [ - w/2.0 ,   l/2.0 ,  - h/2.0 ],
    [ - w/2.0 ,   l/2.0 ,    h/2.0 ],
    [   w/2.0 , - l/2.0 ,  - h/2.0 ],
    [   w/2.0 , - l/2.0 ,    h/2.0 ],
    [   w/2.0 ,   l/2.0 ,  - h/2.0 ],
    [   w/2.0 ,   l/2.0 ,    h/2.0 ]
    ]) 



    vertices_add_rotated = [ Rz.dot(v) for v in vertices_add ]



    base = np.array([
    [ cx, cy, cz ],
    [ cx, cy, cz ],
    [ cx, cy, cz ],
    [ cx, cy, cz ],
    [ cx, cy, cz ],
    [ cx, cy, cz ],
    [ cx, cy, cz ],
    [ cx, cy, cz ]
    ]) 

    vertices = base + vertices_add_rotated



    #angles = np.array([ math.atan2(  math.sqrt ( math.pow(v[0], 2) + math.pow(v[1], 2) , v[2]  ))  for v in vertices ] )
    angles = np.array([ 90 - math.degrees(math.acos(  v[2] / (math.sqrt ( math.pow(v[0], 2) + math.pow(v[1], 2) + math.pow(v[2], 2) )  )) )  for v in vertices ] )




    max_angle = np.amax(angles)
    min_angle = np.amin(angles)


    


    ring_dictionary = {
        15.000 : 29.0,
        10.333: 30,
        7.000 : 25,
        4.6667: 26,
        3.333 : 21,
        2.333 : 22,
        1.667: 17,
        1.333: 13,
        1.000: 18,
        0.667: 14,
        0.333: 9,
        0.000: 5,
        -0.333: 10,
        -0.667: 6,
        -1.000: 1,
        -1.333: 31,
        -1.667: 2,
        -2.000: 28,
        -2.333: 27,
        -2.667: 23,
        -3.000: 24,
        -3.333: 20,
        -3.667: 19,
        -4.000: 15,
        -4.667: 16,
        -5.333: 12,
        -6.148: 11,
        -7.254: 8,
        -8.843: 7,
        -11.310: 4,
        -15.639: 3,
        -25.000: 0
    }

    result = [ v for k,v in ring_dictionary.items() if k <= max_angle and k >= min_angle ]

    if result == []:
        display("loc", loc)
        display("l", l)
        display("w", w)
        display("h", h)
        display("rz", rz)
        display("Rz", Rz)
        display("vertices_add", vertices_add)
        display("vertices_add_rotated", vertices_add_rotated)
        display("vertices", vertices )
        display("angles", angles)
        display("max_angle", max_angle)
        display("min_angle", min_angle)

    #print("changed")


    return [ v for k,v in ring_dictionary.items() if k <= max_angle and k >= min_angle ]




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
        with io.open("/home/dan/Documents/OpenPCDet/data/kitti/training/label_2/" + formatted_idx +  ".txt", mode="r", encoding="utf-8") as f:
            for line in f:
                label = line.strip().split(' ')
                calib = calibration_kitti.Calibration("/home/dan/Documents/OpenPCDet/data/kitti/training/calib/" + formatted_idx +  ".txt")
                h = float(label[8])
                w = float(label[9])
                l = float(label[10])

                loc = np.array((float(label[11]), float(label[12]), float(label[13])), dtype=np.float32)
                loc.reshape(1, 3)
                rz = float(label[14]) # named ry of camera, but its actually rz of lidar

                loc_lidar = calib.rect_to_lidar_daniel(loc)

                if not (get_ring_information(loc_lidar, l, w, h, rz, label[0]) == []):
                    with open("/home/dan/Documents/OpenPCDet/data/kitti/training/label_2_transformed/" + formatted_idx +  ".txt", 'a+') as out_file:
                        out_file.write(line)
                else:
                    display("idx:", idx)

                


                counterdd = counterdd + 1
                print(counterdd)

execute()
