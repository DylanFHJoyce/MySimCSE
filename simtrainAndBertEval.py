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

def makeEmbeddings(datasetName):
  simModel = SimCSE("thisTrainedModel")

  #load dataset to embed
  with open(datasetName, "rb") as f:
    loaded_list = pickle.load(f)
    
  #embed dataset with simcse model 
  embeddings = simModel.encode(loaded_list).numpy()

  #save dataset
  with open("thisModelGeneralEmbeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)


def runBert():
    #DOES THIS NEED SHELL = TRUE ASWELL?
    command = "conda run -n berTopicEnv python runBERTopic.py"

    #test bert will take general embeddings, parameters

    #fit bertopic model to embeddings

    #evaluate and store evaluation

    subprocess.run(command, shell=True)


pd.set_option('display.width', 1000)


#placeholders 
trainingTripletsCSV = "trainingTriplets4000Manual.csv" #
#learning_rate = 5e-5


#USE 5e-4, 2.5e-4, 1e-4, 7.5e-5, 5e-5, 2.5e-5, 1e-5, 7.5e-6
#learningRates = [0, 1e-4, 7.5e-5, 5e-5, 2.5e-5, 1e-5, 7.5e-6]#[5e-4, 1e-4, 5e-5, 1e-5, 5e-6]
#learningRates = [0, 1e-4, 5e-5, 1e-5]# [5e-3, #, 5e-6] #prehaps reverse?
learningRates = [5e-5]
num_epochs = 4
datasetName = "placeholder"



#get the training data and its labels (as a tuple of dataframes train, val, test)
with open('split4000Manual.pkl', 'rb') as f:
  TrainValTest = pickle.load(f)



#clear the folder where result dataframe is stored
#for each learning rate etc
TopicOrder=["learning_rate", "iteration", "topicSize", "Coherence", "TD", "percTrainInMinusOne", "numTopicsGenerated"]
simResults = pd.DataFrame(columns=TopicOrder)
simResults.to_csv("simResults.csv", index=False)



simResults = pd.read_csv("simResults.csv")

#ALSO DO FOR: starting model 
for learning_rate in learningRates:
  
  print("\n\n\n\n\n\n\n\nSTARTING LEARNING RATE", learning_rate)


  #sentence-transformers/all-mpnet-base-v2 (this is the model the bertopic paper uses, but it may be cased)
  startingModel = "princeton-nlp/sup-simcse-bert-base-uncased"
  runSim(startingModel, trainingTripletsCSV, learning_rate, num_epochs)
  
  #makeEmbeddings()
  datasetName = "genDatasetProcessed.pkl"
  #def makeEmbeddings(datasetName):
  simModel = SimCSE("thisTrainedModel")
  
  #load dataset to embed
  with open(datasetName, "rb") as f:
    loaded_list = pickle.load(f)
  #embed dataset with simcse model 
  embeddings = simModel.encode(loaded_list).numpy()
  #save dataset
  with open("thisModelGeneralEmbeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)


  #make embeddings of val data for experiment 2 use
  trainEmbeddings = simModel.encode(TrainValTest[0]["Document"].tolist()).numpy()
  with open("thisModelTrainingEmbeddings.pkl", "wb") as f:
    pickle.dump(trainEmbeddings, f)



  
  #for each bertopic parameters
  
  #COULD HAVE THE PARAM GRID INSIDE OF RUNBERT??
  
  runBert() #runBert loads df and adds its results to it
  
  bertResults = pd.read_csv("bertResults.csv")
  bertResults["learning_rate"] = learning_rate
  
  bertResults = bertResults[TopicOrder]
  simResults = pd.concat([simResults, bertResults], axis=0, ignore_index=True)
  
  print("\n\n\n\n\n\n\n\n\n\n\n\n\nENDING LEARNING RATE", learning_rate)
  print(simResults)
  print("above are all simResults so far")
  
pd.set_option('display.width', 1000)
print(simResults)

simResults.to_csv("simResults.csv", index=False)
print("CHECK IF SHELL=True is needed for subprocesses?")
