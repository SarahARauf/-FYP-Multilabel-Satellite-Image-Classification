import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import os  # Import the os module for handling directories


data_dict = {
    'airplane': 2306, 'freeway': 2500, 'roundabout': 2039,
    'airport': 2480, 'golf course': 2515, 'runway': 2259,
    'bare soil': 39345, 'grass': 49391, 'sand': 11014,
    'baseball diamond': 1996, 'greenhouse': 2601, 'sea': 4980,
    'basketball court': 3726, 'gully': 2413, 'ships': 4092,
    'beach': 2485, 'habor': 2492, 'snow': 3565,
    'bridge': 2772, 'intersection': 2497, 'snowberg': 2555,
    'buildings': 51305, 'island': 2493, 'sparse residential area': 1829,
    'cars': 34013, 'lake': 2499, 'stadium': 2462,
    'chaparral': 5903, 'mobile home': 2499, 'swimming pool': 5078,
    'cloud': 1798, 'mountain': 5468, 'tanks': 2500,
    'containers': 2500, 'overpass': 2652, 'tennis court': 2499,
    'crosswalk': 2673, 'park': 1682, 'terrace': 2345,
    'dense residential area': 2774, 'parking lot': 7061, 'track': 3693,
    'desert': 2537, 'parkway': 2537, 'trail': 12376,
    'dock': 2492, 'pavement': 56383, 'transmission tower': 2500,
    'factory': 2667, 'railway': 4399, 'trees': 70728,
    'field': 15142, 'railway station': 2187, 'water': 27834,
    'football field': 1057, 'river': 2493, 'wetland': 3417,
    'forest': 3562, 'road': 37783, 'wind turbine': 2049
}


data_dict_bigearth2 = {
    "Mixed forest": 211703,
    "Non-irrigated arable land": 196695,
    "Broad-leaved forest": 150944,
    "Complex cultivation patterns": 107786,
    "Water bodies": 83811,
    "Discontinuous urban fabric": 69872,
    "Peatbogs": 23207,
    "Industrial or commercial units": 12895,
    "Olive groves": 12538,
    "Continuous urban fabric": 10784,
    "Vineyards": 9567,
    "Inland marshes": 6236,
    "Sport and leisure facilities": 5353,
    "Mineral extraction sites": 4618,
    "Road and rail networks and associated land": 3384,
    "Green urban areas": 1786,
    "Sparsely vegetated areas": 1563,
    "Coastal lagoons": 1498, 
    "Estuaries": 1086,
    "Airports": 979,
    "Port areas": 509,
    "Burnt areas": 328,
    "Coniferous forest": 211703,
    "Transitional woodland/shrub": 173506,
    "Land principally occupied by agriculture, with significant areas of natural vegetation": 147095,
    "Pastures": 103554,
    "Sea and ocean": 81612,
    "Agro-forestry areas": 30674,
    "Permanently irrigated land": 13589,
    "Natural grassland": 12835,
    "Sclerophyllous vegetation": 11241,
    "Water courses": 10572,
    "Annual crops associated with permanent crops": 7022,
    "Moors and heathland": 5890,
    "Fruit trees and berry plantations": 4754,
    "Rice fields": 3793,
    "Bare rock": 3277,
    "Beaches, dunes, sands": 1578,
    "Salt marshes": 1562,
    "Construction sites": 1174,
    "Intertidal flats": 1003,
    "Dump sites": 959,
    "Salines": 424
    }

data_dict_bigearth = {
    "Complex cultivation patterns": 21470,
    "Land principally occupied by agriculture, with significant areas of natural vegetation": 29427,
    "Broad-leaved forest": 30236,
    "Coniferous forest": 42337,
    "Mixed forest": 43381,
    "Discontinuous urban fabric": 14058,
    "Non-irrigated arable land": 39260,
    "Industrial or commercial units": 2610,
    "Vineyards": 1903,
    "Transitional woodland/shrub": 34568,
    "Peatbogs": 4682,
    "Water bodies": 16853,
    "Sparsely vegetated areas": 303,
    "Sea and ocean": 16309,
    "Permanently irrigated land": 2680,
    "Annual crops associated with permanent crops": 1377,
    "Inland marshes": 1200,
    "Agro-forestry areas": 6062,
    "Sclerophyllous vegetation": 2179,
    "Pastures": 20666,
    "Water courses": 2126,
    "Continuous urban fabric": 2194,
    "Sport and leisure facilities": 1114,
    "Olive groves": 2434,
    "Rice fields": 775,
    "Moors and heathland": 1210,
    "Beaches, dunes, sands": 317,
    "Fruit trees and berry plantations": 943,
    "Green urban areas": 338,
    "Salt marshes": 333,
    "Intertidal flats": 197,
    "Salines": 88,
    "Construction sites": 255,
    "Natural grassland": 2557,
    "Bare rock": 662,
    "Mineral extraction sites": 887,
    "Airports": 183,
    "Coastal lagoons": 301,
    "Estuaries": 220,
    "Road and rail networks and associated land": 666,
    "Port areas": 108,
    "Burnt areas": 66,
    "Dump sites": 189
}


data_dict_metrics = {
    'precision': 'OP',
    'recall': 'OR',
    'f1': 'OF1',
    'loss': 'loss'
}

def Args():
    parser = argparse.ArgumentParser(description="settings")
    # model
    parser.add_argument("--csv_train")
    parser.add_argument("--csv_val")
    parser.add_argument("--csv_test")
    parser.add_argument("--output_dir", default="plots")  # Add an argument for the output directory
    parser.add_argument("--title")  # Add an argument for the output directory
    parser.add_argument("--y_label") 
    parser.add_argument("--x_label", default="epoch")


    args = parser.parse_args()
    return args

# def graph_test_imbalance(df_test, output_dir, title, y_label, x_label):
#     # Map class names to the number of samples
#     num_samples_dict = {class_name: data_dict[class_name] for class_name in df_test['class']}

#     plt.figure(figsize=(12, 8))

#     # Plot precision, recall, and F1 for each class
#     for metric in ['precision', 'recall', 'F1']:
#         plt.plot(num_samples_dict.values(), df_test[metric], label=metric)

#     plt.title(title)
#     plt.xlabel(x_label)
#     plt.ylabel(y_label)
#     plt.legend()
#     plt.grid(True)

#     # Save or show the plot
#     plt.savefig(f"{output_dir}/test_imbalance_plot.png")
#     plt.show()  # Uncomment if you want to display the plot
#     #plt.close()  # Close the plot to free up resources

# def graph_test_imbalance(df_test, output_dir, title, y_label, x_label):
#     # Map class names to the number of samples
#     num_samples_dict = {class_name: data_dict[class_name] for class_name in df_test['class']}

#     # Sort data based on the ascending number of sample sizes
#     sorted_data = sorted(num_samples_dict.items(), key=lambda x: x[1])

#     plt.figure(figsize=(12, 8))

#     # Plot precision, recall, and F1 for each class
#     for metric in ['precision', 'recall', 'F1']:
#         plt.plot([item[1] for item in sorted_data], df_test.loc[df_test['class'].isin([item[0] for item in sorted_data])][metric], label=metric)


#     plt.title(title)
#     plt.xlabel(x_label)
#     plt.ylabel(y_label)
#     plt.legend()
#     plt.grid(True)

#     #plt.xscale('log', base=10)
    

#     # custom_ticks = [value for class_name, value in sorted_data]
#     # plt.xticks(custom_ticks, [str(tick) for tick in custom_ticks], rotation=45, ha="right")

#     custom_ticks = [value for class_name, value in sorted_data]
#     plt.xticks(custom_ticks, [class_name for class_name, _ in sorted_data], rotation=45, ha="right")

#     plt.yticks([i/10 for i in range(11)])


#     # Save or show the plot
#     plt.savefig(f"{output_dir}/test_imbalance_plot.png")
#     plt.show()  # Uncomment if you want to display the plot
#     # plt.close()  # Close the plot to free up resources


def graph_test_imbalance(df_test, output_dir, title, y_label, x_label):

    df_test['sample_size'] = df_test['class'].map(data_dict)

    plt.figure(figsize=(12, 8))

    df_test = df_test.sort_values(by='sample_size', ascending=True)

    sorted_sample_sizes = df_test['sample_size'].astype(str).tolist()

    plt.plot(sorted_sample_sizes, df_test['precision'], label='Precision', marker='o')
    plt.plot(sorted_sample_sizes, df_test['recall'], label='Recall', marker='o')
    plt.plot(sorted_sample_sizes, df_test['F1'], label='F1 Score', marker='o')

    # plt.plot(df_test['sample_size'], df_test['precision'], label='Precision')
    # plt.plot(df_test['sample_size'], df_test['recall'], label='Recall')
    # plt.plot(df_test['sample_size'], df_test['F1'], label='F1 Score')

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)

    #plt.xscale('log', base=10)
    
    plt.xticks(rotation=45)


    # custom_ticks = [value for class_name, value in sorted_data]
    # plt.xticks(custom_ticks, [str(tick) for tick in custom_ticks], rotation=45, ha="right")

    # custom_ticks = [value for class_name, value in sorted_data]
    # plt.xticks(custom_ticks, [class_name for class_name, _ in sorted_data], rotation=45, ha="right")

    plt.yticks([i/10 for i in range(11)])


    # Save or show the plot
    plt.savefig(f"{output_dir}/test_imbalance_plot.png")
    plt.show()  # Uncomment if you want to display the plot
    # plt.close()  # Close the plot to free up resources




def graph_train_val(df_train, df_val, output_dir, title, y_label, x_label):
    x_axis_train = df_train.columns[0]
    x_axis_val = df_val.columns[0]

    # Use only the specified y_label for both training and validation
    y_train = df_train[data_dict_metrics.get(y_label)]
    y_val = df_val[data_dict_metrics.get(y_label)]

    plt.figure(figsize=(10, 6))

    # Plot training and validation data
    plt.plot(df_train[x_axis_train], y_train, label='Training')
    plt.plot(df_val[x_axis_val], y_val, label='Validation')

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)

    # Save or show the plot
    plt.savefig(f"{output_dir}/{title}.png")
    plt.show()  # Uncomment if you want to display the plot
    
    # x_axis = df.columns[0]  # Assuming the first column is the x-axis
    # y_axes = df.columns[1:]  # Assuming the remaining columns are y-axes

    # for y_axis in y_axes:
    #     plt.plot(df[x_axis], df[y_axis], label=y_axis)

    # plt.xlabel(x_label)
    # plt.ylabel(y_label)  # You can customize the ylabel
    # plt.legend()
    # plt.title(title)

    # # Check if the output directory exists, if not, create it
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    # # Save the plot to the specified directory
    # plt.savefig(os.path.join(output_dir, "{}.png".format(title)))
    # plt.show()

def main():
    args = Args()
    if args.csv_train:
        train = pd.read_csv(args.csv_train)
    if args.csv_val:
        val = pd.read_csv(args.csv_val)
    if args.csv_test:
        test = pd.read_csv(args.csv_test)
    
    if args.csv_train and args.csv_val:
        graph_train_val(train, val, args.output_dir, args.title, args.y_label, args.x_label)

    if args.csv_test:
        graph_test_imbalance(test, args.output_dir, args.title, args.y_label, args.x_label)


if __name__ == "__main__":
    main()
