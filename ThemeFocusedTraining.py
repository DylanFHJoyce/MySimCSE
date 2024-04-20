import subprocess
from simcse import SimCSE
import pickle
import numpy as np
import pandas as pd


def runThemeSpreadAnalysis():
    #DOES THIS NEED SHELL = TRUE ASWELL?
    command = "conda run -n berTopicEnv python runThemeSpreadAnalysis.py"

    #test bert will take general embeddings, parameters

    #fit bertopic model to embeddings

    #evaluate and store evaluation

    subprocess.run(command, shell=True)


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
trainLabeledDataDF = trainLabeledDataDF[trainLabeledDataDF["Category"] == "crime"]
print(len(trainLabeledDataDF))



#run training 
#runSim(startingModel, trainingTripletsCSV, learning_rate, num_epochs)



#do bertopic model and spread analysis and compare to starting one 


print("end of theme focused training file")
