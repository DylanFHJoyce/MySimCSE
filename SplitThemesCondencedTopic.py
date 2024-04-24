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


def runSim(startingModel, trainingTripletsCSV, learning_rate, num_epochs, output_dir, per_device_train_batch_size):
  command = (
    "conda run -n simEnv python train.py "
    f"--model_name_or_path {startingModel} "
    f"--train_file {trainingTripletsCSV} "
    f"--output_dir {output_dir} "
    f"--num_train_epochs {num_epochs} "
    f"--per_device_train_batch_size {per_device_train_batch_size} "
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



#have the sections all in here and comment out at whatever stage in the process, its messy but i need to save time.


startingModel = "princeton-nlp/sup-simcse-bert-base-uncased"
print("THIS STARTING MODEL SHOULD BE THE ONE FROM THE OTHER THING WE TRAINED")
#we have a model to start with
#we use it to encode our data etc

datasetName = "genDatasetProcessed.pkl"
#def makeEmbeddings(datasetName):
simModel = SimCSE(startingModel)

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


ThemeFocusedValEmbeddings = simModel.encode(TrainValTest[1]["Document"].tolist()).numpy()
with open("ThemeFocusedValEmbeddings.pkl", "wb") as f:
    pickle.dump(ThemeFocusedValEmbeddings, f)

ThemeFocusedTestEmbeddings = simModel.encode(TrainValTest[2]["Document"].tolist()).numpy()
with open("ThemeFocusedTestEmbeddings.pkl", "wb") as f:
    pickle.dump(ThemeFocusedTestEmbeddings, f)



TopicOrder=["LR", "themeIter", "iteration", "TD", "Coherence", "topicSize", "percTrainInMinusOne", "numTopicsGenerated", "AveMixedMeasure", "percTopicsAreMixed", "percTopicsAreCondenced", "percSpreadThemes", "percCondencedThemes", "aveEnthropy"]
ThemeResults = pd.DataFrame(columns=TopicOrder)
ThemeResults.to_csv("ThemeResults.csv", index=False)
ThemeResults = pd.read_csv("ThemeResults.csv")





#we run a few bert models on it with various parameters, recording the themes that are often deemed condenced by our metrics
#(so our normal spread analysis but we count the times a theme is condenced and add it to the overall count)
#we proceede with one or a few of the most condenced themes

STARTING MODEL (THUS OUTPUT DIR) MUST HAVE "theme" in its name!!!!!!!!!!!
#we run a final bert model to generate a midling amount of topics, we use fit transform to create this model so that we have 
#the topic allocations for the samples in the wider dataset

#we choose one of the topics that has a condenced amount of our target theme
#THIS SHOULD BE A TOPIC WITH ABOVE 1000 SAMPLES FROM THE GENERAL DATASET!
#we run a bertopic model on just the data from that topic, again fit transform to get the different groupings
#we then use this data to generate a triplet dataset


we add our new triplet dataset to the old one and train all together (if we cant work out the other training)
