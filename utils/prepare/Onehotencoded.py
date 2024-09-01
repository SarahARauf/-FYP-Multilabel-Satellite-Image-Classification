
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
import json





# Extract labels and create a set of unique labels
all_labels = [
    "Mixed forest", "Non-irrigated arable land", "Broad-leaved forest", "Complex cultivation patterns", "Water bodies",
    "Discontinuous urban fabric", "Peatbogs", "Industrial or commercial units", "Olive groves", "Continuous urban fabric",
    "Vineyards", "Inland marshes", "Sport and leisure facilities", "Mineral extraction sites", "Road and rail networks and associated land",
    "Green urban areas", "Sparsely vegetated areas", "Coastal lagoons", "Estuaries", "Airports",
    "Port areas", "Burnt areas", "Coniferous forest", "Transitional woodland/shrub", "Land principally occupied by agriculture, with significant areas of natural vegetation", "Pastures",
    "Sea and ocean", "Agro-forestry areas", "Permanently irrigated land", "Natural grassland", "Sclerophyllous vegetation",
    "Water courses", "Annual crops associated with permanent crops", "Moors and heathland", "Fruit trees and berry plantations", "Rice fields", "Bare rock",
    "Beaches, dunes, sands", "Salt marshes", "Construction sites", "Intertidal flats", "Dump sites", "Salines"
]

#  "target": [
#       0, mixed forest
#       0,
#       0,
#       0,
#       0,
#       0,
#       0,
#       0,
#       0,
#       0,
#       0,
#       0,
#       0,
#       0,
#       0,
#       0,
#       0,
#       0,
#       0,
#       1, airport
#       0, pport
#       0, burnt
#       0, conf
#       0, transition
#       1, land princip
#       0, pastures
#       0, sea ocean
#       0, agro
#       0, perm
#       0, natural
#       0, sclerr
#       0, water cour
#       0, annual
#       0, heathland
#       0, fruit trees
#       0, rice field
#       0, bare rock
#       1, beaches sand dunes
#       0, marsh
#       0, constr
#       0, flats
#       0, dump
#       0 saline
#     ],
#     "img_path": "Dataset/BigEarthProcessed//S2A_MSIL2A_20180529T115401_34_84"
#   },

def countlabel(df):

    # Count the occurrences of each class label
    class_counts = Counter()
    for labels in df['class']:
        for label in eval(labels):
            class_counts[label] += 1

    print("Class Label Counts after Downsampling:")
    for label, count in class_counts.items():
        print(f"{label}: {count}")

def traintestsplit(df):

    train_df, val_df = train_test_split(df, random_state=42, test_size=0.3)
    val_df, test_df = train_test_split(val_df, random_state=42, test_size=0.5)
    #trainval_df =pd.concat([train_df, val_df])

    print('train_df:')
    countlabel(train_df)

    
    print('trainval_df:')
    countlabel(val_df)

    print('test_Df:')
    countlabel(test_df)


    print("Test_df len: ",len(test_df))
    print("Train_df len: ",len(train_df))
    print("val len: ", len(val_df))

    return train_df, val_df, test_df


def onehotencoded(df):
    print("start onehotencoded")
    # Create a binary matrix
    one_hot_encoded = pd.DataFrame(0, index=df.index, columns=all_labels)
    one_hot_encoded = pd.concat([df['img_path'], one_hot_encoded], axis=1)
    one_hot_encoded.set_index('img_path', inplace=True)
    #one_hot_encoded.to_csv("testingpurpose.csv", index=False)

    # Populate the matrix with 1s where labels are present
    count=0
    for index, row in df.iterrows():
        count+=1
        #print(count)
        labels_list = eval(row['class'])
        #print(row['img_path'])
        #print(type(row['img_path']))
        one_hot_encoded.loc[row['img_path'], labels_list] = 1

    # Concatenate the original DataFrame with the one-hot encoded matrix
    #result = pd.concat([df['img_path'], one_hot_encoded], axis=1)

    # Save the result to a new CSV file
    #result.to_csv("one_hot_encoded_bigearth.csv", index=False)

    return one_hot_encoded

def to_json(df, name):
    print("start json")
    json_data = []

    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        # Extract the target values (excluding the 'image' column)
        target_values = row[0:].astype(int).tolist()

        # Create a dictionary for the current entry
        entry = {
            "target": target_values,
            "img_path": index
        }

        # Append the entry to the list
        json_data.append(entry)

    # Convert the list to JSON format
    json_output = json.dumps(json_data, indent=2)

    # Save the JSON data to a file in the Colab content space
    with open('{}.json'.format(name), 'w') as json_file:
        json_file.write(json_output)



def main():
        
    # Assuming your .csv file is named 'data.csv'
    file_path = 'downsampled_data.csv'

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    train_df, val_df, test_df = traintestsplit(df)

    #test_df.to_csv("test.csv", index=False)



    train_df = onehotencoded(train_df)
    val_df = onehotencoded(val_df)
    test_df = onehotencoded(test_df)


    #test_df.to_csv("test_onehot.csv", index=True)
    #test_df.to_csv("test.csv", index=False)

    to_json(train_df, "train_bigearth")
    to_json(val_df, "val_bigearth")
    to_json(test_df, "test_bigearth")






main()
