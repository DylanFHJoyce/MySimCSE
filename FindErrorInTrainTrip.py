import subprocess
from simcse import SimCSE
import pickle
import numpy as np
import pandas as pd
import random
#change training data file

def generate_triplet_dataset(input_df, length):
    # lists for triplet data
    sent0_list, sent1_list, hard_neg_list = [], [], []

    for _ in range(length):
        # Randomly select a row from the input DataFrame
        random_row_index = random.randint(0, len(input_df) - 1)
        sent0_row = input_df.iloc[random_row_index]

        # Randomly select a row from the same category as sent0
        same_category_rows = input_df[input_df['Category'] == sent0_row['Category']]
        sent1_row = same_category_rows.sample(1).iloc[0]

        # Randomly select a row from a different category than sent0
        different_category_rows = input_df[input_df['Category'] != sent0_row['Category']]
        hard_neg_row = different_category_rows.sample(1).iloc[0]

        # Append the selected rows to the lists
        sent0_list.append(sent0_row['Document'])
        sent1_list.append(sent1_row['Document'])
        hard_neg_list.append(hard_neg_row['Document'])

    # Create the triplet DataFrame
    triplet_df = pd.DataFrame({
        'sent0': sent0_list,
        'sent1': sent1_list,
        'hard_neg': hard_neg_list
    })

    return triplet_df


def runSim(startingModel, trainingTripletsCSV, learning_rate, num_epochs):
  command = (
    "conda run -n simEnv python train.py "
    f"--model_name_or_path {startingModel} "
    f"--train_file {trainingTripletsCSV} "
    "--output_dir thisTrainedModel "
    f"--num_train_epochs {num_epochs} "
    "--per_device_train_batch_size 64 "
    f"--learning_rate {learning_rate} "
    "--max_seq_length 64 "
    "--load_best_model_at_end "
    "--pooler_type cls "
    "--overwrite_output_dir "
    "--temp 0.05 "
     "--do_train "
     "--fp16 "
     "--use_in_batch_instances_as_negatives"
  )
  subprocess.run(command, shell=True)


# #open  the laelled data (format train, val, test)
with open('split4000Manual.pkl', 'rb') as f:
    TrainValTest = pickle.load(f)
trainLabeledDataDF = TrainValTest[0]



specificThemeTripletDataset = generate_triplet_dataset(trainLabeledDataDF, 250)
specificThemeTripletDataset.to_csv("altTripFindError.csv", index=False)


startingModel = "princeton-nlp/sup-simcse-bert-base-uncased"

# ourTripTrain = pd.read_csv("specificThemeTripletDataset.csv")
# print(len(ourTripTrain))
# altTripFindError = ourTripTrain.iloc[100:180, :]
#altTripFindError.to_csv("altTripFindError.csv", index=False)


themeFocusModel = "themeFocusModel"
trainingTripletsCSV = "altTripFindError.csv"
learning_rate =5e-5
runSim(startingModel, trainingTripletsCSV, learning_rate, 1)#, themeFocusModel)
