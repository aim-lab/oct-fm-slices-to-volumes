import os
import pandas as pd
import numpy as np


def describe_dataset(dataset, name_dataset):

    
    if hasattr(dataset, "patients") and hasattr(dataset, "get_label_counts"):
        label_counts = dataset.get_label_counts()
        count_pos, count_neg = label_counts[1], label_counts[0]
        #TODO: change following line with dataset.get_patient_list()
        print(f"In the {name_dataset}, there is {len(dataset)} images for {len(set(dataset.patients))} patients, and we have {(count_pos/len(dataset) * 100):.0f}% BT+ and {(count_neg/len(dataset) * 100):.0f}% BT-")
        return dataset.get_patient_list()
    else:
        print(f"In the {name_dataset}, there is {len(dataset)} images")

def save_patients_split(train_patients, val_patients, test_patients, path_save):
    """
    Save patients split in a csv file.
    """

    train_df = pd.DataFrame(train_patients, columns=["patient_id"])
    val_df = pd.DataFrame(val_patients, columns=["patient_id"])
    test_df = pd.DataFrame(test_patients, columns=["patient_id"])
    train_df["set"] = "train"
    val_df["set"] = "val"
    test_df["set"] = "test"

    if not os.path.isdir(os.path.dirname(path_save)):
        os.mkdir(os.path.dirname(path_save))
    print("SAVING patient split")
    pd.concat([train_df, val_df, test_df]).to_csv(path_save, index=False)

def get_idx_train_val(path_patient_split, list_patients): #it's going to be training_set.list_patients
    train_val_split = pd.read_csv(path_patient_split) 
    training_patients = set(train_val_split[train_val_split["set"]=="train"]['patient_id'])
    valid_patients = set(train_val_split[train_val_split["set"]=="val"]['patient_id'])

    idx_train = np.where([patient in training_patients for patient in list_patients])[0]
    idx_val = np.where([patient in valid_patients for patient in list_patients])[0]

    return idx_train, idx_val