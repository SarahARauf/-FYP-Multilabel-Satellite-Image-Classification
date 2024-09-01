
import csv #added
import os #added
import  pandas as pd
import numpy as np
import json
import os
import gc
from glob import glob
import matplotlib.pyplot as plt
import rasterio as rio
import cv2
from skimage import io, exposure, img_as_uint, img_as_float
from itertools import product


#from rasterio.plot import show


# B00: Coastal aerosol; 60m
# B01: Blue; 10m
# B02: Green; 10m
# B03: Red; 10m
# B04: Vegetation red edge; 20m
# B05: Vegetation red edge; 20m
# B06: Vegetation red edge; 20m
# B07: NIR; 10m
# B08: Water vapor; 60m
# B09: SWIR; 20m
# B10: SWIR; 20m
# B11: Narrow NIR; 20m

#False colour (urban) shortwave infrared (swir2), swir1, red
#color infrared (vegetation) NIR, red, green
#agriculture swir1, nir, blue
#land/water nir, swir1, red
#natural with atmospheric removal swir2, nir, green
#shortwave infrared swir2, nir, red
#vegetation analysis swir1, nir,red
map_colour = {'rgb': [3, 2, 1],
              'fc': [10,9,3],
              'ir': [7, 3, 2],
              'agri': [9,7,1],
              'lw': [7,9,3],
              'nwar': [10, 7,2],
              'sir': [10,7,3],
              'veg':[9,7,3]}

COUNT = 0


def mergeImg3band(names, new_folder, img_type):
    global COUNT  # Declare COUNT as global


    S_sentinel_bands = glob(names+"/*B?*.tif")
    S_sentinel_bands.sort()

    l = []

    for i in S_sentinel_bands:
        with rio.open(i, 'r') as f:
            l.append(f.read(1))

    resize_l = []
    for img in l:
        resize_img = cv2.resize(img,  (120, 120))
        resize_l.append(resize_img)

    rgb_l = [resize_l[i] for i in map_colour.get(img_type)]

    #rgb_l = l[1:4]
    arr_st = np.stack(rgb_l)

    #arr_st = ((arr_st /(np.max(arr_st))) * 255).astype(np.uint8)


    arr_st = (((arr_st - np.min(arr_st)) /(np.max(arr_st) - np.min(arr_st)))).astype(np.float32)


    #file_path = os.path.join(new_folder, f'{names.split("\\")[1]}.jpg')
    # print("this is the name: ",names)
    # print("this is the datatype: ",type(names))

    file_path = os.path.join(new_folder, names.split("/")[1] +f"_{img_type}"+".jpg")




    plt.imsave(file_path, np.copy(arr_st.transpose((1, 2, 0))))
    print(COUNT,"img saved to:", file_path)
    #print(arr_st.shape)



def main():
    global COUNT  # Declare COUNT as global

    file_path = 'downsampled_data.csv'

    #Load the dataset
    df = pd.read_csv(file_path)


    # Extract the last part of the path for each img_path
    df['img_path'] = df['img_path'].apply(lambda x: "BigEarthNet-v1.0/" + x.split('//')[-1])

    # Print the updated DataFrame with only the last part of the path
    names = df['img_path'].tolist()

    #print(names)

    #folder_path = "BigEarthNet-v1.0/"
    new_folder = "E:/Downloads/BigEarthProcessed2/"



    #gets all the folder names from bigEarthNet
    #names = glob("BigEarthNet-v1.0/*", recursive = True)

    #[print(n,"\n") for n in names] #BigEarthNet-v1.0\S2A_MSIL2A_20170617T113321_84_56

    #[print(n.split("\\")[1],"\n") for n in names] #S2A_MSIL2A_20170617T113321_84_56




    # if os.path.exists(new_folder):
    #     os.rmdir(new_folder)


   # mergeImg3band(names[300000], new_folder)
    # mergeImg3band("BigEarthNet-v1.0\S2B_MSIL2A_20180417T102019_64_65", new_folder, "rgir")
    # mergeImg3band("BigEarthNet-v1.0\S2A_MSIL2A_20170613T101031_10_58", new_folder, "vre")

    # for key in map_colour.keys():

    #     mergeImg3band("BigEarthNet-v1.0\S2A_MSIL2A_20170613T101031_10_58", new_folder, key)


    for paths in names:
        # for key in map_colour.keys():
        COUNT += 1 
        mergeImg3band(paths, new_folder, 'rgb')
        mergeImg3band(paths, new_folder, 'fc')
        mergeImg3band(paths, new_folder, 'ir')
        mergeImg3band(paths, new_folder, 'agri')
        mergeImg3band(paths, new_folder, 'lw')
        mergeImg3band(paths, new_folder, 'nwar')
        mergeImg3band(paths, new_folder, 'sir')
        mergeImg3band(paths, new_folder, 'veg')

    # for paths, key in product(names, map_colour.keys()):
    #     mergeImg3band(paths, new_folder, key)





    # for n in names:
    #     mergeImg3band(n, new_folder)









#     data = json.load(open(folder_path+folder_path[:-1]+'_labels_metadata.json'))
#     print(data)


# S_sentinel_bands = glob(folder_path+"/*B?*.tif")
# S_sentinel_bands.sort()

main()

# if __name__ == "__main__":
#     main()







