import subprocess
from simcse import SimCSE
import pickle
import numpy as np
import pandas as pd
#change training data file

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


startingModel = "princeton-nlp/sup-simcse-bert-base-uncased"

ourTripTrain = pd.read_csv("specificThemeTripletDataset.csv")
print(len(ourTripTrain))
altTripFindError = ourTripTrain.iloc[100:200, :]

altTripFindError.to_csv("altTripFindError.csv", index=False)


themeFocusModel = "themeFocusModel"
trainingTripletsCSV = "altTripFindError.csv"
learning_rate =5e-5
runSim(startingModel, trainingTripletsCSV, learning_rate, 1)#, themeFocusModel)
