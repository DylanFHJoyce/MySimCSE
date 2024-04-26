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
    "--overwrite_output_dir " #removing this did not work
    "--temp 0.05 "
     "--do_train "
     "--fp16 "
     "--use_in_batch_instances_as_negatives"
  )
  subprocess.run(command, shell=True)


def getTopIdxs(df, column_name, x):
    topIdxs = df.nlargest(x, column_name).index
    return topIdxs


def getBottomIdxs(df, column_name, x):
    bottomIdxs = df.nsmallest(x, column_name).index
    return bottomIdxs




# print("STARTING?")

# output_dir = "themeFocusModel" #if changing this change further up in file aswell (test ver)
# trainingTripletsCSV = "specificThemeTripletDataset.csv"
# learning_rates = [0, 1e-4, 5e-5, 5e-6]
# per_device_train_batch_size = 64 #CHANGE THIS IF USING LOWER QUANTITIES OF TRAINING DATA OR DUPLICATE TRAINING DATA

# runSim(output_dir, trainingTripletsCSV, 5e-5, 2, output_dir, per_device_train_batch_size)





#use either base model or sim model to start
#sentence-transformers/all-mpnet-base-v2 (this is the model the bertopic paper uses, but it may be cased)
#output_dir = "themeFocusModel"


####################################################################################################
#get embeddings for starting model
startingModel = "princeton-nlp/sup-simcse-bert-base-uncased" #This has randomly stopped working?
#startingModel = "google-bert/bert-base-uncased"
#startingModel = "sentence-transformers/all-mpnet-base-v2"
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
####################################################################################################




###############################################################################################
#dataframe for bertopic model results
TopicOrder=["LR", "epoch", "iteration", "TD", "Coherence", "topicSize", "percTrainInMinusOne", "numTopicsGenerated", "AveMixedMeasure", "percTopicsAreMixed", "percTopicsAreCondenced", "percSpreadThemes", "percCondencedThemes", "aveEnthropy"]
ThemeResults = pd.DataFrame(columns=TopicOrder)
ThemeResults.to_csv("ThemeResults.csv", index=False)
ThemeResults = pd.read_csv("ThemeResults.csv")
#############################################################################################



ThemesToFocusBASEMODELDF = pd.read_csv("ThemesToFocusBASEMODELDF.csv", index_col=0)
print("\n\nHERE ARE THE THEMES TO FOCUS FROM BASE MODEL RESULTS\n", ThemesToFocusBASEMODELDF)

topIEntropy = getTopIdxs(ThemesToFocusBASEMODELDF, "enthropy", 3)
bottomIEntropy = getBottomIdxs(ThemesToFocusBASEMODELDF, "enthropy", 3)


print("HERE IS topIEntropy: ", topIEntropy)
print("HERE IS bottomIEntropy: ", bottomIEntropy)

print("HERE IS topIEntropy: ", type(topIEntropy))
print("HERE IS topIEntropy: ", type(topIEntropy.tolist()))
print("HERE IS topIEntropy: ", topIEntropy.tolist())

#get results for base model to decide 
# print("STARTING FIRST THEME SPREAD ANALYSIS")
# runThemeSpreadAnalysis()
###################### ################# ############################## ################### ############################ HERE!


# topI = getTopIdxs(ThemesToFocusDF, "enthropy", 3)
# bottomi = getBottomIdxs(ThemesToFocusDF, "enthropy", 3)

# print("HERE IS TOPI: ", topI)
# print("HERE IS bottomi: ", bottomi)










#turn labelled training data into triplet dataset based on theme (keep small percentage of general data to keep context)
trainLabeledDataDF = TrainValTest[0]



#allThemes = trainLabeledDataDF["Category"].value_counts()
allThemes = trainLabeledDataDF["Category"].unique().tolist()
print("ALL THEMES", allThemes)
themeBasedTriplets = {}
themeSamplesMultiplier = {theme: 1.0 for theme in allThemes}
print(themeSamplesMultiplier)
print("AAAA\n\n\n\n")
for theme in allThemes:
    print(theme, "\n")
    focusCategories = [theme]
    print(focusCategories, "\nIS THIS RIGHT?")

    thisThemeTrainLabeledDataDFFocus = trainLabeledDataDF[trainLabeledDataDF["Category"].isin(focusCategories)]
    thisThemeTrainLabeledDataDFFocus.reset_index(drop=True)

    print(focusCategories)
    print(len(thisThemeTrainLabeledDataDFFocus))
    
    allOtherThemeTrainLabeledDataDFNonFocus = trainLabeledDataDF[~trainLabeledDataDF["Category"].isin(focusCategories)]
    focusSamples = len(thisThemeTrainLabeledDataDFFocus)
    percentFromNonFocus = 0.2


    print(len(allOtherThemeTrainLabeledDataDFNonFocus))
    random_indices = np.random.choice(allOtherThemeTrainLabeledDataDFNonFocus.index, int(focusSamples * percentFromNonFocus), replace=False)
    allOtherThemeTrainLabeledDataDFNonFocus = allOtherThemeTrainLabeledDataDFNonFocus.loc[random_indices]
    allOtherThemeTrainLabeledDataDFNonFocus.reset_index(drop=True)
    
    # print(len(trainLabeledDataDFFocus))
    
    thisThemeFocusAndPercentOfNonFocusDf = pd.concat([thisThemeTrainLabeledDataDFFocus, allOtherThemeTrainLabeledDataDFNonFocus])
    #thisThemeFocusAndPercentOfNonFocusDf.to_csv("thisThemeFocusAndPercentOfNonFocusDf.csv")

    thisThemeTripletDataset = generate_triplet_dataset(thisThemeFocusAndPercentOfNonFocusDf, 500)

    themeBasedTriplets[theme] = thisThemeTripletDataset

#print(themeBasedTriplets)
#THEN 
#FOR theme in themeBasedTriplets (for key, value in?)
    #make blank training df
    #training dataframe = itself concat x samples from theme
    #x could be 200 * y with y staring at 1.0 and being stored in a dict that can be increaced/decreaced based
    #on spread each iteration
#then save as the thing we are to use


concThemeTriplets = pd.DataFrame()
for theme, value in themeBasedTriplets.items():
    numSamples = int(200 * themeSamplesMultiplier[theme])
    print(numSamples)
    concThemeTriplets = pd.concat([concThemeTriplets, value.head(numSamples)])
concThemeTriplets.reset_index(drop=True, inplace=True)
print("MUST SHUFFLE THIS DATASET BEFORE USING IT FOR TRAINING!")
concThemeTriplets = concThemeTriplets.sample(frac=1).reset_index(drop=True)
concThemeTriplets.to_csv("concThemeTriplets.csv", index=False)

print(concThemeTriplets)
#print(concThemeTriplets)

#print(len(specificThemeTripletDataset))




#print(thisThemeTripletDataset)


##

# #MUST HAVE AT LEAST ONE THEME OMITTED FOR THE OTHER PART TO WORK (OR CHANGE THIS NEXT SECTION TO SKIP IF THERE ISNT)
# focusCategories = ["crime", "Discrimination/representation/rights", "protest/public concern"]
# focusCategories = ["crime"]

focusCategories = topIEntropy.tolist()
trainLabeledDataDFFocus = trainLabeledDataDF[trainLabeledDataDF["Category"].isin(focusCategories)]
trainLabeledDataDFFocus.reset_index(drop=True)

print("LEN OF TRAINING DATA FROM THOSE THINGS: ", len(trainLabeledDataDFFocus))
# #trainLabeledDataDFFocus = pd.concat([trainLabeledDataDFFocus] * 20, ignore_index=True)

trainLabeledDataDFNonFocus = trainLabeledDataDF[~trainLabeledDataDF["Category"].isin(focusCategories)]
focusSamples = len(trainLabeledDataDFFocus)
percentFromNonFocus = 0.2

print("\n\n\nTHIS IS THE LENGTH BEING MADE INTO SAMPLES")
print(int(focusSamples/percentFromNonFocus))
print("\n\n\n")

# #take random sample of NonFocus df to keep general context

random_indices = np.random.choice(trainLabeledDataDFNonFocus.index, int(focusSamples * percentFromNonFocus), replace=False)
trainLabeledDataDFNonFocus = trainLabeledDataDFNonFocus.loc[random_indices]
trainLabeledDataDFNonFocus.reset_index(drop=True)

# print(len(trainLabeledDataDFFocus))

FocusAndPercentOfNonFocusDf = pd.concat([trainLabeledDataDFFocus, trainLabeledDataDFNonFocus])
FocusAndPercentOfNonFocusDf.to_csv("FocusAndPercentOfNonFocusDf.csv")




#COULD INCREACE LEN OF DATA GENERATED IF ERRORS PERSIST
specificThemeTripletDataset = generate_triplet_dataset(FocusAndPercentOfNonFocusDf, 4000)# len(FocusAndPercentOfNonFocusDf))

#BELOW WAS THE ONE USED FOR ALL DATA
#specificThemeTripletDataset = generate_triplet_dataset(trainLabeledDataDF, 4000)


#specificThemeTripletDataset = generate_triplet_dataset(FocusAndPercentOfNonFocusDf, 200)
print(len(specificThemeTripletDataset))
specificThemeTripletDataset.to_csv("specificThemeTripletDataset.csv", index=False)








#run training 
#need to save triplet set and then feed it in as runSim gets it by file name not by internal parameter
output_dir = "themeFocusbertModel" #if changing this change further up in file aswell (test ver)
#output_dir = "mybertModel"
trainingTripletsCSV = "specificThemeTripletDataset.csv"
learning_rates = [5e-5]#2.5e-5]#[1.5e-4, 3e-4]#2.5e-5, 7.5e-5]#5e-5, 5e-6] #0, 1e-4, done already
per_device_train_batch_size = 64 #CHANGE THIS IF USING LOWER QUANTITIES OF TRAINING DATA OR DUPLICATE TRAINING DATA

print("firstTrain")
for learning_rate in learning_rates: #for x in range(0, 11, 2):
    #for ThemeFocusedIteration in range(5, 41, 10): #THEN CHANGE TO 26 AND START AT 16
    for ThemeFocusedIteration in range(0, 2):
        ThemeResults = pd.read_csv("ThemeResults.csv")
        #startingModel = output_dir runThemeSpreadAnalysis()
        #STARTING MODEL (THUS OUTPUT DIR) MUST HAVE "theme" in its name!!!!!!!!!!!
        print("\n\n\n\n\n\n\nSTARTINGMODEL", startingModel, "\n\n\n\n\n\n\n")
        #startingModel = output_dir
        print("LR STARTING: ", learning_rate)
        #output_dir = "bertout"
        #runSim(startingModel, trainingTripletsCSV, learning_rate, 6, output_dir, per_device_train_batch_size)
        runSim(startingModel, trainingTripletsCSV, learning_rate, 4, output_dir, per_device_train_batch_size)
        trainingTripletsCSV = "concThemeTriplets.csv"
        #startingModel = output_dir #after first training run we use that model for each subsequent run
        #trainingTripletsCSV = "SECONDspecificThemeTripletDataset"
        #learning_rate = 5e-6
        
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
        ThemeSpreadAnalysisBertResults["epoch"] = ThemeFocusedIteration
        ThemeSpreadAnalysisBertResults["LR"] = learning_rate
        
        ThemeSpreadAnalysisBertResults = ThemeSpreadAnalysisBertResults[TopicOrder]
        ThemeResults = pd.concat([ThemeResults, ThemeSpreadAnalysisBertResults], axis=0, ignore_index=True)
        
        pd.set_option('display.width', 1000)
        print(ThemeResults)
        print("above are all ThemeResults so far")
        
        ThemeResults.to_csv("ThemeResults.csv", index=False)












#do bertopic model and spread analysis and compare to starting one 


print("end of theme focused training file")
