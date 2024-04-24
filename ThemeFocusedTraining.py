import subprocess
from simcse import SimCSE
import pickle
import numpy as np
import pandas as pd

import random

print("\n\n\nSTARTING THEME FOCUSED TRAINING: ")

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
    "--tokenizer_name bert-base-uncased "
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



# print("STARTING?")

# output_dir = "themeFocusModel" #if changing this change further up in file aswell (test ver)
# trainingTripletsCSV = "specificThemeTripletDataset.csv"
# learning_rates = [0, 1e-4, 5e-5, 5e-6]
# per_device_train_batch_size = 64 #CHANGE THIS IF USING LOWER QUANTITIES OF TRAINING DATA OR DUPLICATE TRAINING DATA

# runSim(output_dir, trainingTripletsCSV, 5e-5, 2, output_dir, per_device_train_batch_size)





#use either base model or sim model to start
#sentence-transformers/all-mpnet-base-v2 (this is the model the bertopic paper uses, but it may be cased)
#output_dir = "themeFocusModel"
#startingModel = "princeton-nlp/sup-simcse-bert-base-uncased" #This has randomly stopped working?
startingModel = "google-bert/bert-base-uncased"

#startingModel = "sentence-transformers/all-mpnet-base-v2"


#do embeddings
#makeEmbeddings()
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





#do bert model and use theme spread analysis to decide upon themes to train





print("STARTING FIRST THEME SPREAD ANALYSIS")
runThemeSpreadAnalysis()






#turn labelled training data into triplet dataset based on theme (keep small percentage of general data to keep context)
trainLabeledDataDF = TrainValTest[0]


#MUST HAVE AT LEAST ONE THEME OMITTED FOR THE OTHER PART TO WORK (OR CHANGE THIS NEXT SECTION TO SKIP IF THERE ISNT)
focusCategories = ["crime", "Discrimination/representation/rights", "protest/public concern"]
focusCategories = ["crime"]

trainLabeledDataDFFocus = trainLabeledDataDF[trainLabeledDataDF["Category"].isin(focusCategories)]
trainLabeledDataDFFocus.reset_index(drop=True)

#trainLabeledDataDFFocus = pd.concat([trainLabeledDataDFFocus] * 20, ignore_index=True)



trainLabeledDataDFNonFocus = trainLabeledDataDF[~trainLabeledDataDF["Category"].isin(focusCategories)]
focusSamples = len(trainLabeledDataDFFocus)
percentFromNonFocus = 0.1


print("\n\n\nTHIS IS THE LENGTH BEING MADE INTO SAMPLES")
print(int(focusSamples/percentFromNonFocus))
print("\n\n\n")


#take random sample of NonFocus df to keep general context

random_indices = np.random.choice(trainLabeledDataDFNonFocus.index, int(focusSamples * percentFromNonFocus), replace=False)
trainLabeledDataDFNonFocus = trainLabeledDataDFNonFocus.loc[random_indices]
trainLabeledDataDFNonFocus.reset_index(drop=True)

print(len(trainLabeledDataDFFocus))


FocusAndPercentOfNonFocusDf = pd.concat([trainLabeledDataDFFocus, trainLabeledDataDFNonFocus])

FocusAndPercentOfNonFocusDf.to_csv("FocusAndPercentOfNonFocusDf.csv")


#COULD INCREACE LEN OF DATA GENERATED IF ERRORS PERSIST

#specificThemeTripletDataset = generate_triplet_dataset(FocusAndPercentOfNonFocusDf, 4000)# len(FocusAndPercentOfNonFocusDf))
specificThemeTripletDataset = generate_triplet_dataset(trainLabeledDataDF, 4000)

#specificThemeTripletDataset = generate_triplet_dataset(FocusAndPercentOfNonFocusDf, 200)
print(len(specificThemeTripletDataset))
specificThemeTripletDataset.to_csv("specificThemeTripletDataset.csv", index=False)












#run training 
#need to save triplet set and then feed it in as runSim gets it by file name not by internal parameter
output_dir = "themeFocusModel" #if changing this change further up in file aswell (test ver)
#output_dir = "mybertModel"
trainingTripletsCSV = "specificThemeTripletDataset.csv"
learning_rates = [5e-5]#[1.5e-4, 3e-4]#2.5e-5, 7.5e-5]#5e-5, 5e-6] #0, 1e-4, done already
per_device_train_batch_size = 64 #CHANGE THIS IF USING LOWER QUANTITIES OF TRAINING DATA OR DUPLICATE TRAINING DATA

print("firstTrain")
for learning_rate in learning_rates:
    for ThemeFocusedIteration in range(0, 1): #DONT CHANGE THIS, we do multiple iters anyway in the bertopic process
        ThemeResults = pd.read_csv("ThemeResults.csv")
        #startingModel = output_dir
        #STARTING MODEL (THUS OUTPUT DIR) MUST HAVE "theme" in its name!!!!!!!!!!!
        print("\n\n\n\n\n\n\nSTARTINGMODEL", startingModel, "\n\n\n\n\n\n\n")
        runSim(startingModel, trainingTripletsCSV, learning_rate, 4, output_dir, per_device_train_batch_size)
        
        startingModel = output_dir #after first training run we use that model for each subsequent run
    
        
        #redo Embeddings with new focus model
        simModel = output_dir
        
        # datasetName = "genDatasetProcessed.pkl"
        # #def makeEmbeddings(datasetName):
        simModel = SimCSE(simModel)
        
        
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
        
        
        # #do bert model and use theme spread analysis to decide upon themes to train
        
        
        print("\n\nSTARTING SECOND THEME SPREAD ANALYSIS/ secondTrain")
        runThemeSpreadAnalysis()
    
    
    
    
        #####################YOU WOULD ALSO DO THIS AFTER THE BASE MODEL RUN
        ThemeSpreadAnalysisBertResults = pd.read_csv("ThemeSpreadAnalysisBertResults.csv")
        ThemeSpreadAnalysisBertResults["themeIter"] = ThemeFocusedIteration
        ThemeSpreadAnalysisBertResults["LR"] = learning_rate
        
        ThemeSpreadAnalysisBertResults = ThemeSpreadAnalysisBertResults[TopicOrder]
        ThemeResults = pd.concat([ThemeResults, ThemeSpreadAnalysisBertResults], axis=0, ignore_index=True)
        
        pd.set_option('display.width', 1000)
        print(ThemeResults)
        print("above are all ThemeResults so far")
        
        ThemeResults.to_csv("ThemeResults.csv", index=False)












#do bertopic model and spread analysis and compare to starting one 


print("end of theme focused training file")
