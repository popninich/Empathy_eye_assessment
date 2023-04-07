import numpy as np
import pandas as pd
import math
import glob
from tqdm import tqdm
import os
import shutil

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT_FOLDER = "."
SEED = 42

# Define used 26 columns from the total 71 columns in the input file 
INPUT_COLS = [
    "Participant name",
    "Project name",
    "Eyetracker timestamp",
    "Event", "Event value",
    "Gaze point X", 
    "Gaze point Y",
    "Gaze point left X", "Gaze point left Y", 
    "Gaze point right X", "Gaze point right Y", 
    "Gaze direction left X", "Gaze direction left Y",
    "Gaze direction left Z", "Gaze direction right X",
    "Gaze direction right Y", "Gaze direction right Z",
    "Pupil diameter left", "Pupil diameter right", 
    "Validity left", "Validity right",
    "Eye movement type", "Gaze event duration",
    "Eye movement type index", 
    "Fixation point X", "Fixation point Y"
]

FEATURE_COLS = [
    "avg_change_pupil_left",
    "avg_change_pupil_right",
    "avg_change_total_pupil",
    "peak_pupil_left",
    "peak_pupil_right",
    "peak_total_pupil",
    "short_fixation_count",
    "medium_fixation_count",
    "long_fixation_count",
    "avg_fixation_duration",
    "peak_fixation_duration",
    "saccadic_count",
    "avg_saccadic_durations",
    "avg_saccadic_velocity",
    "peak_saccadic_velocity",
]

TARGET_COL = "qcae_score_after"


def _prepare_pupil_diameter(value):
    if value != value:
        # Check NaN value
        return value
    
    if isinstance(value, float):
        return value
    
    # Change string of x,xx to float x.xx
    # Because the dataset use the comma for a decimal separator
    return float(value.replace(",", "."))

def _keep_test_files(files, participant, ratio=0.2):    
    train_files = files[:int(len(files)*(1-ratio))]
    test_files = [f for f in files if f not in train_files]
    # Save train and test files
    train_filepath = os.path.join(ROOT_FOLDER, "data", "train", participant)
    test_filepath = os.path.join(ROOT_FOLDER, "data", "test", participant)
    if not os.path.exists(train_filepath):
        os.makedirs(train_filepath)
    if not os.path.exists(test_filepath):
        os.makedirs(test_filepath)
    for f in train_files:
        filename = f.split("/")[-1]
        shutil.copyfile(f, os.path.join(train_filepath, filename))
    for f in test_files:
        filename = f.split("/")[-1]
        shutil.copyfile(f, os.path.join(test_filepath, filename))
    return train_files

# Prepare dataset
def _prepare_data():
    data_filepath = os.path.join(
        ROOT_FOLDER, 
        "data", 
        "EyeT", 
        "*.csv"
    )
    data_files = {}

    # Combine participant's trials into a list for its participant
    for file_name in sorted(glob.glob(data_filepath)):    
        splits = file_name.split(".csv")[0].split("_")
        
        dataset = splits[3]
        participant = splits[9]
        trial_number = splits[11]
        
        if participant not in data_files:
            data_files[participant] = []
        data_files[participant].append([int(trial_number), dataset, file_name])
            
    # Sort files by trial number for every participant
    for key, value in data_files.items():
        data_files[key] = sorted(value)
        
    # Define labels
    target_before_df = pd.read_csv(
        os.path.join(
            ROOT_FOLDER, 
            "data", 
            "label", 
            "Questionnaire_datasetIA.csv"
        ), 
        encoding="cp1252" # Reading files from Windows-1252
    )

    target_after_df = pd.read_csv(
        os.path.join(
            ROOT_FOLDER, 
            "data", 
            "label", 
            "Questionnaire_datasetIB.csv"
        ), 
        encoding="cp1252"
    )

    target_before_df = target_before_df[["Participant nr", "Total Score extended"]].set_index("Participant nr")
    target_after_df = target_after_df[["Participant nr", "Total Score extended"]].set_index("Participant nr")

    qcae_score_before = target_before_df.to_dict("index")
    qcae_score_after = target_after_df.to_dict("index")
    
    # Prepare training data
    all_participant_data = []
    for participant, values in tqdm(data_files.items(), desc="Training loop"):
        # Keep 20% to be test set
        train_files = _keep_test_files([v[2] for v in values], participant)
        dataset = values[0][1]
        df = pd.read_csv(
            train_files[0],
            low_memory=False # We don't want to specific dtype so we use the low_memory instead
        )
        # Append data for the participant
        for file in train_files[1:]:
            tmp_df = pd.read_csv(
                file,
                low_memory=False
            )
            df = pd.concat([df, tmp_df], ignore_index=True)
        df = df[INPUT_COLS]
        df = df[df["Event"].isnull()].reset_index()

        # Pupil size features
        df["Pupil diameter left"] = df.apply(lambda row: _prepare_pupil_diameter(row["Pupil diameter left"]), axis=1)
        df["Pupil diameter right"] = df.apply(lambda row: _prepare_pupil_diameter(row["Pupil diameter right"]), axis=1)
        df["Pupil diameter"] = df.apply(lambda row: row["Pupil diameter left"] + row["Pupil diameter right"], axis=1)
        pupil_diameter_lefts = df["Pupil diameter left"].dropna().values
        pupil_diameter_rights = df["Pupil diameter right"].dropna().values
        pupil_diameters = df["Pupil diameter"].dropna().values
        change_pupil_lefts = np.diff(pupil_diameter_lefts)
        change_pupil_rights = np.diff(pupil_diameter_rights)
        change_pupil = np.diff(pupil_diameters)
        avg_change_pupil_left = np.mean(change_pupil_lefts)
        avg_change_pupil_right = np.mean(change_pupil_rights)
        avg_change_total_pupil = np.mean(change_pupil)
        peak_pupil_left = np.max(pupil_diameter_lefts)
        peak_pupil_right = np.max(pupil_diameter_rights)
        peak_total_pupil = np.max(pupil_diameters)

        # Unclassified or eye-not-found values
        total_data_points = df.shape[0]
        unk_data_points = df[(df["Eye movement type"] == "Unclassified") | (df["Eye movement type"] == "EyesNotFound")].shape[0]

        # Fixation features
        fixation_durations = []
        fixation_index = []
        for idx, row in df.iterrows():
            if row["Eye movement type"] == "Fixation":
                if row["Eye movement type index"] not in fixation_index:
                    fixation_index.append(row["Eye movement type index"])
                    fixation_durations.append(row["Gaze event duration"])
        short_fixation_count = len([i for i in fixation_durations if i < 150])
        medium_fixation_count = len([i for i in fixation_durations if i >= 150 and i <= 900])
        long_fixation_count = len([i for i in fixation_durations if i > 900])
        avg_fixation_duration = np.mean(fixation_durations)
        peak_fixation_duration = np.max(fixation_durations)
        
        # Saccade features
        saccade_durations = []
        saccade_index = []
        saccade_velocity = []
        diff_gaze_x = np.diff(df["Gaze point X"])
        diff_gaze_y = np.diff(df["Gaze point Y"])
        diff_time = np.diff(df["Eyetracker timestamp"])/1000 # To seconds
        start_saccade = False
        for idx, row in df.iterrows():
            if row["Eye movement type"] == "Saccade":
                if row["Eye movement type index"] not in saccade_index:
                    saccade_index.append(row["Eye movement type index"])
                    saccade_durations.append(row["Gaze event duration"])
                if not start_saccade: 
                    start_saccade = True
                    continue
                else:
                    if idx > 0:
                        # In order to calculate the gaze velocity in degrees per second
                        # We will have to convert the data from its cartesian coordinate system to a spherical coordinate system
                        curr_x = row["Gaze point X"]
                        curr_y = row["Gaze point Y"]
                        curr_t = row["Eyetracker timestamp"]
                        prev_x = df.iloc[idx-1, 6]
                        prev_y = df.iloc[idx-1, 7]
                        prev_t = df.iloc[idx-1, 3]
                        vec1 = [curr_x, curr_y]
                        vec2 = [prev_x, prev_y]
                        dot_product = np.dot(vec1, vec2)
                        prod_of_norms = np.linalg.norm(vec1) * np.linalg.norm(vec2)
                        angle = np.degrees(np.arccos(dot_product / prod_of_norms))
                        diff_t = (curr_t-prev_t)/1000 # Milliseconds to seconds
                        if math.isnan(dot_product) or math.isnan(angle):
                            continue
                        saccade_velocity.append(angle/diff_t)
            else:
                start_saccade = False
                
        saccadic_count = len(saccade_index)
        avg_saccadic_durations = np.mean(saccade_durations)
        avg_saccadic_velocity = np.mean(saccade_velocity)
        peak_saccadic_velocity = np.max(saccade_velocity)
        
        # Insert features
        row = {
            "participant": participant,
            "task": "visual communication" if dataset == "III" else "foraging visual",
            "total_data_points": total_data_points,
            "unk_data_points": unk_data_points,
            "avg_change_pupil_left": avg_change_pupil_left,
            "avg_change_pupil_right": avg_change_pupil_right,
            "avg_change_total_pupil": avg_change_total_pupil,
            "peak_pupil_left": peak_pupil_left,
            "peak_pupil_right": peak_pupil_right,
            "peak_total_pupil": peak_total_pupil,
            "short_fixation_count": short_fixation_count,
            "medium_fixation_count": medium_fixation_count,
            "long_fixation_count": long_fixation_count,
            "avg_fixation_duration": avg_fixation_duration,
            "peak_fixation_duration": peak_fixation_duration,
            "saccadic_count": saccadic_count,
            "avg_saccadic_durations": avg_saccadic_durations,
            "avg_saccadic_velocity": avg_saccadic_velocity,
            "peak_saccadic_velocity": peak_saccadic_velocity,
            "qcae_score_before": qcae_score_before[int(participant)]["Total Score extended"],
            "qcae_score_after": qcae_score_after[int(participant)]["Total Score extended"]
        }
        all_participant_data.append(row)
    # Save to data file for the feature exploration
    data_filename = os.path.join(ROOT_FOLDER, "data", "data_df.csv")
    final_data_df = pd.DataFrame.from_dict(all_participant_data)
    final_data_df.to_csv(data_filename, header=True, index=False)
    
    
    # Test data
    test_data_files = {}
    test_filepath = os.path.join(ROOT_FOLDER, "data", "test", "*", "*.csv")

    for file_name in sorted(glob.glob(test_filepath)):    
        splits = file_name.split(".csv")[0].split("_")
        
        dataset = splits[3]
        participant = splits[9]
        trial_number = splits[11]
        
        if participant not in test_data_files:
            test_data_files[participant] = []
        test_data_files[participant].append([int(trial_number), dataset, file_name])
            
    # Sort files by trial number for every participant
    for key, value in test_data_files.items():
        test_data_files[key] = sorted(value)
        
    test_participant_data = []
    for participant, values in tqdm(test_data_files.items(), desc="Test loop"):
        test_files = [v[2] for v in values]
        dataset = values[0][1]
        df = pd.read_csv(
            test_files[0],
            low_memory=False
        )
        for file in test_files[1:]:
            tmp_df = pd.read_csv(
                file,
                low_memory=False
            )
            df = pd.concat([df, tmp_df], ignore_index=True)
        df = df[INPUT_COLS]
        df = df[df["Event"].isnull()].reset_index()

        # Pupil size features
        df["Pupil diameter left"] = df.apply(lambda row: _prepare_pupil_diameter(row["Pupil diameter left"]), axis=1)
        df["Pupil diameter right"] = df.apply(lambda row: _prepare_pupil_diameter(row["Pupil diameter right"]), axis=1)
        df["Pupil diameter"] = df.apply(lambda row: row["Pupil diameter left"] + row["Pupil diameter right"], axis=1)
        pupil_diameter_lefts = df["Pupil diameter left"].dropna().values
        pupil_diameter_rights = df["Pupil diameter right"].dropna().values
        pupil_diameters = df["Pupil diameter"].dropna().values
        change_pupil_lefts = np.diff(pupil_diameter_lefts)
        change_pupil_rights = np.diff(pupil_diameter_rights)
        change_pupil = np.diff(pupil_diameters)
        avg_change_pupil_left = np.mean(change_pupil_lefts)
        avg_change_pupil_right = np.mean(change_pupil_rights)
        avg_change_total_pupil = np.mean(change_pupil)
        peak_pupil_left = np.max(pupil_diameter_lefts)
        peak_pupil_right = np.max(pupil_diameter_rights)
        peak_total_pupil = np.max(pupil_diameters)

        # Unclassified or eye-not-found values
        total_data_points = df.shape[0]
        unk_data_points = df[(df["Eye movement type"] == "Unclassified") | (df["Eye movement type"] == "EyesNotFound")].shape[0]

        # Fixation features
        fixation_durations = []
        fixation_index = []
        for idx, row in df.iterrows():
            if row["Eye movement type"] == "Fixation":
                if row["Eye movement type index"] not in fixation_index:
                    fixation_index.append(row["Eye movement type index"])
                    fixation_durations.append(row["Gaze event duration"])
        short_fixation_count = len([i for i in fixation_durations if i < 150])
        medium_fixation_count = len([i for i in fixation_durations if i >= 150 and i <= 900])
        long_fixation_count = len([i for i in fixation_durations if i > 900])
        avg_fixation_duration = np.mean(fixation_durations)
        peak_fixation_duration = np.max(fixation_durations)
        
        # Saccade features
        saccade_durations = []
        saccade_index = []
        saccade_velocity = []
        diff_gaze_x = np.diff(df["Gaze point X"])
        diff_gaze_y = np.diff(df["Gaze point Y"])
        diff_time = np.diff(df["Eyetracker timestamp"])/1000 # To seconds
        start_saccade = False
        for idx, row in df.iterrows():
            if row["Eye movement type"] == "Saccade":
                if row["Eye movement type index"] not in saccade_index:
                    saccade_index.append(row["Eye movement type index"])
                    saccade_durations.append(row["Gaze event duration"])
                if not start_saccade: 
                    start_saccade = True
                    continue
                else:
                    if idx > 0:
                        # In order to calculate the gaze velocity in degrees per second
                        # We will have to convert the data from its cartesian coordinate system to a spherical coordinate system
                        curr_x = row["Gaze point X"]
                        curr_y = row["Gaze point Y"]
                        curr_t = row["Eyetracker timestamp"]
                        prev_x = df.iloc[idx-1, 6]
                        prev_y = df.iloc[idx-1, 7]
                        prev_t = df.iloc[idx-1, 3]
                        vec1 = [curr_x, curr_y]
                        vec2 = [prev_x, prev_y]
                        dot_product = np.dot(vec1, vec2)
                        prod_of_norms = np.linalg.norm(vec1) * np.linalg.norm(vec2)
                        angle = np.degrees(np.arccos(dot_product / prod_of_norms))
                        diff_t = (curr_t-prev_t)/1000 # Milliseconds to seconds
                        if math.isnan(dot_product) or math.isnan(angle):
                            continue
                        saccade_velocity.append(angle/diff_t)
            else:
                start_saccade = False
                
        saccadic_count = len(saccade_index)
        avg_saccadic_durations = np.mean(saccade_durations)
        avg_saccadic_velocity = np.mean(saccade_velocity)
        peak_saccadic_velocity = np.max(saccade_velocity)
        
        # Insert features
        row = {
            "participant": participant,
            "task": "visual communication" if dataset == "III" else "foraging visual",
            "total_data_points": total_data_points,
            "unk_data_points": unk_data_points,
            "avg_change_pupil_left": avg_change_pupil_left,
            "avg_change_pupil_right": avg_change_pupil_right,
            "avg_change_total_pupil": avg_change_total_pupil,
            "peak_pupil_left": peak_pupil_left,
            "peak_pupil_right": peak_pupil_right,
            "peak_total_pupil": peak_total_pupil,
            "short_fixation_count": short_fixation_count,
            "medium_fixation_count": medium_fixation_count,
            "long_fixation_count": long_fixation_count,
            "avg_fixation_duration": avg_fixation_duration,
            "peak_fixation_duration": peak_fixation_duration,
            "saccadic_count": saccadic_count,
            "avg_saccadic_durations": avg_saccadic_durations,
            "avg_saccadic_velocity": avg_saccadic_velocity,
            "peak_saccadic_velocity": peak_saccadic_velocity,
            "qcae_score_before": qcae_score_before[int(participant)]["Total Score extended"],
            "qcae_score_after": qcae_score_after[int(participant)]["Total Score extended"]
        }
        test_participant_data.append(row)
    # Save to data file for the test purpose
    test_data_filename = os.path.join(ROOT_FOLDER, "data", "test_data_df.csv")
    final_test_data_df = pd.DataFrame.from_dict(test_participant_data)
    final_test_data_df.to_csv(test_data_filename, header=True, index=False)
    
    
if __name__ == '__main__':
    # Check if dataset and label folders exist
    if not os.path.exists(os.path.join(ROOT_FOLDER, "data", "EyeT")):
        raise FileNotFoundError(f"Expected a dataset to be in a folder data/EyeT, but it does not exist.")
    if not os.path.exists(os.path.join(ROOT_FOLDER, "data", "label")):
        raise FileNotFoundError(f"Expected a label to be in a folder data/label, but it does not exist.")
    
    # Create training and test data folders if not exists
    if not os.path.exists(os.path.join(ROOT_FOLDER, "data", "train")):
        os.makedirs(os.path.join(ROOT_FOLDER, "data", "train"))
    if not os.path.exists(os.path.join(ROOT_FOLDER, "data", "test")):
        os.makedirs(os.path.join(ROOT_FOLDER, "data", "test"))
    _prepare_data()
    print(f"---------------- Finish data preparation! ----------------")