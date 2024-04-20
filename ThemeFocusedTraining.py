import subprocess
from simcse import SimCSE
import pickle
import numpy as np
import pandas as pd

import random

def runThemeSpreadAnalysis():
    #DOES THIS NEED SHELL = TRUE ASWELL?
    command = "conda run -n berTopicEnv python runThemeSpreadAnalysis.py"

    #test bert will take general embeddings, parameters

    #fit bertopic model to embeddings

    #evaluate and store evaluation

    subprocess.run(command, shell=True)

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


def runSim(startingModel, trainingTripletsCSV, learning_rate, num_epochs, output_dir):
  command = (
    "conda run -n simEnv python train.py "
    f"--model_name_or_path {startingModel} "
    f"--train_file {trainingTripletsCSV} "
    f"--output_dir {themeFocusModel} "
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




#use either base model or sim model to start
#sentence-transformers/all-mpnet-base-v2 (this is the model the bertopic paper uses, but it may be cased)
startingModel = "princeton-nlp/sup-simcse-bert-base-uncased"

#do embeddings
#makeEmbeddings()
datasetName = "genDatasetProcessed.pkl"
#def makeEmbeddings(datasetName):
simModel = SimCSE("thisTrainedModel")

#load dataset to embed
with open(datasetName, "rb") as f:
  loaded_list = pickle.load(f)
#embed dataset with simcse model 
ThemeSpreadEmbeddings = simModel.encode(loaded_list).numpy()

with open("ThemeSpreadEmbeddings.pkl", "wb") as f:
    pickle.dump(ThemeSpreadEmbeddings, f)



#open  the laelled data (format train, val, test)
with open('split4000Manual.pkl', 'rb') as f:
    TrainValTest = pickle.load(f)
#make embeddings of val data for experiment 2 use
ThemeFocusedTrainingEmbeddings = simModel.encode(TrainValTest[0]["Document"].tolist()).numpy()
with open("ThemeFocusedTrainingEmbeddings.pkl", "wb") as f:
    pickle.dump(ThemeFocusedTrainingEmbeddings, f)



#do bert model and use theme spread analysis to decide upon themes to train


runThemeSpreadAnalysis()



#turn labelled training data into triplet dataset based on theme (keep small percentage of general data to keep context)
trainLabeledDataDF = TrainValTest[0]

focusCategory = "crime"

trainLabeledDataDFFocus = trainLabeledDataDF[trainLabeledDataDF["Category"] == focusCategory]
trainLabeledDataDFNonFocus = trainLabeledDataDF[trainLabeledDataDF["Category"] != focusCategory]
focusSamples = len(trainLabeledDataDFFocus)
percentFromNonFocus = 0.1

#take random sample of NonFocus df to keep general context
random_indices = np.random.choice(trainLabeledDataDFNonFocus.index, int(focusSamples/percentFromNonFocus), replace=False)
trainLabeledDataDFNonFocus = trainLabeledDataDFNonFocus.loc[random_indices]

print(len(trainLabeledDataDFFocus))


FocusAndPercentOfNonFocusDf = pd.concat([trainLabeledDataDFFocus, trainLabeledDataDFNonFocus])

specificThemeTripletDataset = generate_triplet_dataset(FocusAndPercentOfNonFocusDf, len(FocusAndPercentOfNonFocusDf))
specificThemeTripletDataset.to_csv("specificThemeTripletDataset.csv", index=False)

#run training 
#need to save triplet set and then feed it in as runSim gets it by file name not by internal parameter
themeFocusModel = "themeFocusModel"
runSim(startingModel = startingModel, trainingTripletsCSV = "specificThemeTripletDataset.csv", learning_rate =5e-5, num_epochs = 4, output_dir = themeFocusModel)


#redo Embeddings with new focus model

#do bertopic model and spread analysis and compare to starting one 


print("end of theme focused training file")