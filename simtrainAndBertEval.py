import subprocess
from simcse import SimCSE
import pickle
import numpy as np
#change training data file

def runSim(trainingTripletsCSV, learning_rate, num_epochs):
  command = (
    "conda run -n simEnv python train.py "
    "--model_name_or_path princeton-nlp/sup-simcse-bert-base-uncased "
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





#placeholders 
trainingTripletsCSV = "trainingTriplets4000Manual.csv"
learning_rate = 5e-5
#learningRates = [5e-1, 5e-2, 5e-3, 5e-4, 5e-5, 5e-6] #prehaps reverse?
num_epochs = 5
datasetName = "placeholder"



#clear the folder where result dataframe is stored
#for each learning rate etc
#for learning_rate in learningRates:

runSim(trainingTripletsCSV, learning_rate, num_epochs)

makeEmbeddings(datasetName = "genDatasetProcessed.pkl")




#for each bertopic parameters

#COULD HAVE THE PARAM GRID INSIDE OF RUNBERT??

runBert() #runBert loads df and adds its results to it




print("CHECK IF SHELL=True is needed for subprocesses?")
