import json
from glob import glob
import pandas as pd


def createDataFrame2():

    # Creating an empty DataFrame
    df = pd.DataFrame(columns=["img_path", "class"])
    
    img_paths = glob("BigEarthNet-v1.0/*", recursive = True)

    new_path = "Dataset/BigEarthProcessed/"
    #new path =  "Dataset/MLRSNETdevkit\\mlrsnet/JPEGImages/eroded_farmland_01183.jpg"


    for i in img_paths:
        
        img_name = i.split("\\")[1]
        print(img_name)

        data = json.load(open(i+"\\"+img_name+'_labels_metadata.json'))

        # Extract labels
        labels = data.get("labels", [])
        #print(labels)
        # Create one-hot encoding
        #one_hot = [1 if label in labels else 0 for label in class_labels]
        # Add row to DataFrame
        df.loc[len(df)] = [new_path+"/"+img_name] + [labels]

    

   # Displaying the DataFrame
    return df










# def createDataFrame():
#     # Define your class labels
#     class_labels = [
#         "Mixed forest", "Non-irrigated arable land", "Broad-leaved forest", 
#         "Complex cultivation patterns", "Water bodies", "Discontinuous urban fabric", 
#         "Peatbogs", "Industrial or commercial units", "Olive groves", 
#         "Continuous urban fabric", "Vineyards", "Inland marshes", 
#         "Sport and leisure facilities", "Mineral extraction sites", 
#         "Road and rail networks and associated land", "Green urban areas", 
#         "Sparsely vegetated areas", "Coastal lagoons", "Estuaries", "Airports", 
#         "Port areas", "Burnt areas", "Coniferous forest", 
#         "Transitional woodland/shrub", "Agricultural with Natural Vegetation", 
#         "Pastures", "Sea and ocean", "Agro-forestry areas", 
#         "Permanently irrigated land", "Natural grassland", "Sclerophyllous vegetation", 
#         "Water courses", "Permanent crops", "Moors and heathland", 
#         "Fruit trees and berry plantations", "Rice fields", "Bare rock", 
#         "Beaches, dunes, sands", "Salt marshes", "Construction sites", 
#         "Intertidal flats", "Dumpsites", "Salines"
#     ]

#     # Creating an empty DataFrame
#     df = pd.DataFrame(columns=["img_path"] + class_labels)
    
#     img_paths = glob("BigEarthNet-v1.0/*", recursive = True)

#     new_path = "Dataset/BigEarthProcessed/"
#     #new path =  "Dataset/MLRSNETdevkit\\mlrsnet/JPEGImages/eroded_farmland_01183.jpg"


#     for i in img_paths:
        
#         img_name = i.split("\\")[1]
#         print(img_name)

#         data = json.load(open(i+"\\"+img_name+'_labels_metadata.json'))

#         # Extract labels
#         labels = data.get("labels", [])
#         # Create one-hot encoding
#         one_hot = [1 if label in labels else 0 for label in class_labels]
#         # Add row to DataFrame
#         df.loc[len(df)] = [new_path+"/"+img_name] + one_hot

#    # Displaying the DataFrame
#     df.head()

def main():
    df = createDataFrame2()
    df.to_csv("BigEarthNetLabels.csv", index=False)

if __name__ == "__main__":
    main()
