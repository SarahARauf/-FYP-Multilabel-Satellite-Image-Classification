import argparse
import torch
import numpy as np
import json
import csv
from tqdm import tqdm
from .cal_mAP import json_map
from .cal_PR import json_metric, metric, json_metric_top3
from .confusion_matrix import generate_confusion_matrix  # adjust the import based on the actual script name



voc_classes = ("aeroplane", "bicycle", "bird", "boat", "bottle",
           "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor")
coco_classes = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
               'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
               'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
               'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush')
wider_classes = (
                "Male","longHair","sunglass","Hat","Tshiirt","longSleeve","formal",
                "shorts","jeans","longPants","skirt","faceMask", "logo","stripe")

mlrsnet_classes = ('airplane', 'airport', 'bare soil', 'baseball diamond',
       'basketball court', 'beach', 'bridge', 'buildings', 'cars', 'chaparral',
       'cloud', 'containers', 'crosswalk', 'dense residential area', 'desert',
       'dock', 'factory', 'field', 'football field', 'forest', 'freeway',
       'golf course', 'grass', 'greenhouse', 'gully', 'habor', 'intersection',
       'island', 'lake', 'mobile home', 'mountain', 'overpass', 'park',
       'parking lot', 'parkway', 'pavement', 'railway', 'railway station',
       'river', 'road', 'roundabout', 'runway', 'sand', 'sea', 'ships', 'snow',
       'snowberg', 'sparse residential area', 'stadium', 'swimming pool',
       'tanks', 'tennis court', 'terrace', 'track', 'trail',
       'transmission tower', 'trees', 'water', 'wetland', 'wind turbine')

bigearth_classes = ("Mixed forest", "Non-irrigated arable land", "Broad-leaved forest", "Complex cultivation patterns", "Water bodies",
    "Discontinuous urban fabric", "Peatbogs", "Industrial or commercial units", "Olive groves", "Continuous urban fabric",
    "Vineyards", "Inland marshes", "Sport and leisure facilities", "Mineral extraction sites", "Road and rail networks and associated land",
    "Green urban areas", "Sparsely vegetated areas", "Coastal lagoons", "Estuaries", "Airports",
    "Port areas", "Burnt areas", "Coniferous forest", "Transitional woodland/shrub", "Land principally occupied by agriculture, with significant areas of natural vegetation", "Pastures",
    "Sea and ocean", "Agro-forestry areas", "Permanently irrigated land", "Natural grassland", "Sclerophyllous vegetation",
    "Water courses", "Annual crops associated with permanent crops", "Moors and heathland", "Fruit trees and berry plantations", "Rice fields", "Bare rock",
    "Beaches, dunes, sands", "Salt marshes", "Construction sites", "Intertidal flats", "Dump sites", "Salines")



class_dict = {
    "voc07": voc_classes,
    "coco": coco_classes,
    "wider": wider_classes,
    "mlrsnet": mlrsnet_classes,
    "bigearth": bigearth_classes
}



# def evaluation(result, types, ann_path, results_csv, epoch, is_testing_data=False): #add a flag here testing data = false
#     print("Evaluation")
#     classes = class_dict[types]
#     aps = np.zeros(len(classes), dtype=np.float64)

#     ann_json = json.load(open(ann_path, "r"))
#     pred_json = result

#     for i, _ in enumerate(tqdm(classes)): 
#         ap = json_map(i, pred_json, ann_json, types) #calculates for each class
#         aps[i] = ap #need to save this.
#     OP, OR, OF1, CP, CR, CF1, precision_list, recall_list, f1_list  = json_metric(pred_json, ann_json, len(classes), types)
#     mAP = np.mean(aps)
#     print("mAP: {:4f}".format(mAP))
#     print("CP: {:4f}, CR: {:4f}, CF1 :{:4F}".format(CP, CR, CF1))
#     print("OP: {:4f}, OR: {:4f}, OF1 {:4F}".format(OP, OR, OF1))

#     #added
#     with open(results_csv, mode='a', newline='') as file:
#         fieldnames = ['epoch', 'mAP', 'CP', 'CR', 'CF1', 'OP', 'OR', 'OF1']
#         writer = csv.DictWriter(file, fieldnames=fieldnames)

#         #create a dictionary with the metrics
#         metrics = {
#             'epoch': epoch,
#             'mAP': mAP,
#             'CP': CP,
#             'CR': CR,
#             'CF1': CF1,
#             'OP': OP,
#             'OR': OR,
#             'OF1': OF1
#         }

#         #write the metrics to the CSV file
#         writer.writerow(metrics)
#     #for class (should put a flag so that this doesn't run during training)
#     if is_testing_data:
#         with open(is_testing_data, mode='a', newline='') as file:
#             fieldnames =  ['class','precision', 'recall', 'F1']
#             writer = csv.DictWriter(file, fieldnames=fieldnames)

#             for i, class_name in enumerate(classes):
#                 metrics = {
#                     'class': class_name,
#                     'precision': precision_list[i],
#                     'recall': recall_list[i],
#                     'F1': f1_list[i],
#                 }
#                 writer.writerow(metrics)

#             # Write the metrics to the CSV file
#             writer.writerow(metrics)
#     # Confusion matrix generation
#     # Inside your evaluation function or script
#     confusion_matrix_file = f"checkpoint/test/confusion_matrix.png"
#     generate_confusion_matrix(predictions_path=test_results, ann_path=test_file[0], class_names=','.join(classes), output_path=confusion_matrix_file)


def evaluation(result, types, ann_path, results_csv, epoch, is_testing_data=False, test_results=None, test_file=None): 
    print("Evaluation")
    classes = class_dict[types]
    aps = np.zeros(len(classes), dtype=np.float64)

    # Load annotations and predictions
    ann_json = json.load(open(ann_path, "r"))
    pred_json = result

    # Calculate average precision for each class
    for i, _ in enumerate(tqdm(classes)): 
        ap = json_map(i, pred_json, ann_json, types)
        aps[i] = ap

    # Calculate other metrics
    OP, OR, OF1, CP, CR, CF1, precision_list, recall_list, f1_list = json_metric(pred_json, ann_json, len(classes), types)
    mAP = np.mean(aps)
    print("mAP: {:4f}".format(mAP))
    print("CP: {:4f}, CR: {:4f}, CF1 :{:4F}".format(CP, CR, CF1))
    print("OP: {:4f}, OR: {:4f}, OF1 {:4F}".format(OP, OR, OF1))

    # Write metrics to CSV
    with open(results_csv, mode='a', newline='') as file:
        fieldnames = ['epoch', 'mAP', 'CP', 'CR', 'CF1', 'OP', 'OR', 'OF1']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        metrics = {
            'epoch': epoch,
            'mAP': mAP,
            'CP': CP,
            'CR': CR,
            'CF1': CF1,
            'OP': OP,
            'OR': OR,
            'OF1': OF1
        }
        writer.writerow(metrics)

    # Write class-wise metrics if in testing mode
    if is_testing_data:
        with open(is_testing_data, mode='a', newline='') as file:
            fieldnames = ['class', 'precision', 'recall', 'F1']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            for i, class_name in enumerate(classes):
                metrics = {
                    'class': class_name,
                    'precision': precision_list[i],
                    'recall': recall_list[i],
                    'F1': f1_list[i],
                }
                writer.writerow(metrics)

    # Confusion matrix generation
    if test_results and test_file:
        confusion_matrix_file = f"checkpoint/{types}/confusion_matrix.png"  # Adjust path as needed
        generate_confusion_matrix(predictions_path=test_results, ann_path=test_file[0], class_names=','.join(classes), output_path=confusion_matrix_file)
