print("MADE IT TO runBERTopic")

from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
import pickle
import numpy as np
import pandas as pd
import os


import gensim
from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
import nltk
from nltk.tokenize import word_tokenize



#EXPERIMENT 2 IMPORTS



pd.set_option('display.width', 1000)

#nltk.download('punkt')

print("first need to calculate corpus")
def NPMICoherence(topicModel, tokenized_corpus, corpusDictionary):


    #remove -1 from topics (-1 is those that didnt fall into a topic)
    topicReps = topicModel.get_topic_info()[1:]["Representation"].tolist()
    # Calculate NPMI coherence
    coherence_model_npmi = gensim.models.CoherenceModel(topics=topicReps, texts=tokenized_corpus, dictionary=corpusDictionary, coherence='c_npmi', topn=3) #or <10
    # # Calculate NPMI coherence
    #coherence_model_npmi = gensim.models.CoherenceModel(topics=topicModel.get_topic_info()["Representation"].tolist(), texts=tokenized_corpus, dictionary=corpusDictionary, coherence='c_npmi'would put top n here but changing back)
    
    coherence_npmi = coherence_model_npmi.get_coherence()
    
    coherence_npmiPerTopic = coherence_model_npmi.get_coherence_per_topic()
    
    print("NPMI Coherence Score:", coherence_npmi)
    
    return coherence_npmi#, coherence_npmiPerTopic
    
def topicDiversity(topicsNWordsList): #you give this bertopicmodel.get_topic_info()["Representation"].tolist()
    
    uniqueSet = set()
    for topicNWords in topicsNWordsList:
        
        thisTopicSet = set(topicNWords)
        uniqueSet = uniqueSet.union(thisTopicSet)
    
    return len(uniqueSet) / (10 *len(topicsNWordsList))
    
def convertTopicNumToName(numList, df):
    """"
    takes:
    list of topic number predictions
    df from bertModel.get_topic_info() for the bert model that made the predictions

    returns topic (by name not number) for each list entry

    """
    topicDict = dict(zip(df['Topic'], df['Name']))
    return [topicDict.get(topic_number, 'Unknown') for topic_number in numList]

def getTopicPredByName(bertModel, trainDocData, docDataEmbedded):
    """"
    takes:
    already fitted bertopic model,
    df of documents with at least "Document" column, for which we want to see predictions,
    embeddings for the documents

    #transforms documents to the model and retrieves topic number predictions
    #converts topic number predictions to topic name predictions

    returns labelled training df with topic name predicitons attatched

    """
    
    trainDocData = trainDocData.reset_index(drop=True)
    trainDocs = trainDocData["Document"]
    #if type(docDataEmbedded) == torch.Tensor:
        #docDataEmbedded = docDataEmbedded.numpy()

    print("tempCHange in get top pred name")
    #bertModel.transform(trainDocs, docDataEmbedded) #
    pred, prob = bertModel.transform(trainDocData, docDataEmbedded)

    topicNamePredictions = convertTopicNumToName(pred, bertModel.get_topic_info())

    #print(len(topicNamePredictions), len(trainDocData))

    trainDocData["BertPredictions"] = topicNamePredictions

    return trainDocData

def compareTrainTopicsToBTopics(berTopicModel, trainingDataDF, trainingDataEmbeddings):
    '''
    fitted bert model
    training data df with columns for document and category
    embeddings for those documents

    returns crosstab (freq table) for how many times samples from each theme ended up in each topic
    
    '''
    TPbyName = getTopicPredByName(berTopicModel, trainingDataDF, trainingDataEmbeddings)
    
    return pd.crosstab(TPbyName["Category"], TPbyName["BertPredictions"])

def statsFromCrosstab(crosstab):
    #print("for predictions frequency if two catas have high perc in same BT T analyse")

    #for each training category what bertopic topic did its samples end up in most
    most_common_predictions = crosstab.idxmax(axis=1)

    # for each trainging Category Calculate the percentage that ended up in each BertPrediction 
    #THIS IS X% (as 0.X) of label cata ended up in B topic
    prediction_frequency = crosstab.div(crosstab.sum(axis=1), axis=0)
    #and this is the inverse: X% of this B topic came from this label cata
    BTTrainComp = BTopicPercentageTrainComposition(crosstab)

    # Calculate the total count of samples for each BertPrediction
    total_samples_per_prediction = crosstab.sum(axis=0)
    # Calculate the composition of each BertPrediction in terms of Category
    prediction_composition = crosstab.div(total_samples_per_prediction, axis=1)

    #find the categorys that are most spread out across BertPredictions
    category_spread = prediction_frequency.std(axis=1)
    average_category_spread = category_spread.mean()

    #find the least spread Categorys
    least_spread_categories = category_spread.nsmallest(5)  # Adjust the number as needed

    # find the MOST spread Categorys
    most_spread_categories = category_spread.nlargest(5)  # Adjust the number as needed

    return most_common_predictions, prediction_frequency, total_samples_per_prediction, prediction_composition, average_category_spread, category_spread, least_spread_categories, most_spread_categories, BTTrainComp

def BTopicPercentageTrainComposition(crosstab):
    '''
    X% of this B topic came from this label cata
    '''
    newDF = pd.DataFrame(index = crosstab.index)
    for BertTopicGenTopic in crosstab.columns: # for each bertopic topic
        totalSamples = crosstab[BertTopicGenTopic].sum()
        relevance_percentage = (crosstab[BertTopicGenTopic] / totalSamples) * 100
        #print(relevance_percentage)
        newDF[BertTopicGenTopic] = relevance_percentage
    return newDF

def topicsToThemes(BTTrainComp):
    ''' 
    
    give BTTrainComp from statsFromCrosstab
    
    #unseenGeneralDatasetDocs for merge

    converts bert gen topics to themes based on training (or held out) data content
    '''

    #for each bertopic topic generate empty dict entry
    bertTopicCorrelateDict = {key: [] for key in BTTrainComp.columns.values.tolist()}
    
    #assign each topic to the theme that it contains the most samples from (held out data)
    for bertGenTopic in BTTrainComp.columns:
        topicBelongsToTheme = BTTrainComp[bertGenTopic].idxmax()
        bertTopicCorrelateDict.update({bertGenTopic: topicBelongsToTheme})
        
    #for bertTopicCorrelateDict where value is still empty
    #find non empty/most similar topic on same level of hierarchy / adjacent / above?
    #merge it

    print("but if the theme contains no samples... for all dict entries that are still empty: start at lowest hierarchy layer?")


    return bertTopicCorrelateDict


#take general embeddings, parameters

#get the proceessed general dataset list
with open("genDatasetProcessed.pkl", "rb") as f:
  generalDataset = pickle.load(f)
  

#get the general dataset embeddings that were made after the simmodel was trained
with open("thisModelGeneralEmbeddings.pkl", "rb") as f:
  generalEmbeddings = pickle.load(f)
  print("GENSET TYPE", type(generalDataset))



############## NEW COMMIT
#get the tokenized corpus and dictionary made from the gen dataset (including labelled data to have an accurate count)
with open('labelInclusiveTokenizedCorpusAndDictionary.pkl', 'rb') as f:
    corpusAndDictionaryLabelInc = pickle.load(f)
########################

#get the training data and its labels (as a tuple of dataframes train, val, test)
with open('split4000Manual.pkl', 'rb') as f:
    TrainValTest = pickle.load(f)


#ALSO LOAD THE EMBEDDING MODEL #DONT NEED TO BECAUSE EMBEDDINGS ARE PRECOMPUTED?
generalEmbeddings = np.load("thisModelGeneralEmbeddings.pkl", allow_pickle=True)
#and provide params here


#load val data embeddings
with open("thisModelTrainingEmbeddings.pkl", "rb") as f:
    thisModelTrainingEmbeddings = pickle.load(f)

  





#with additional params

print("COULD HAVE THE PARAM GRID HERE TO ITERATE THROUGH, ADDITING TO RESULTS DF EACH TIME")
print("have one df, BERTRESULTS, that is added to each time then taken into simresults overalldf in sim script")

#make and save new bertResults df

bertResults = pd.DataFrame(columns=["iteration", "TD", "Coherence", "topicSize", "percTrainInMinusOne", "numTopicsGenerated"])
bertResults.to_csv("bertResults.csv", index=False)

bertResults = pd.read_csv("bertResults.csv")

topicSizes = [20, 40, 60, 80, 100]#, 60, 80, 100]
topicSizes = [20, 60, 100]
#
print("MIN TOPIC SIZE CHANGED TO NR_TOPICS")
for min_topic_size in topicSizes:
    for iteration in range(0, 2):
        
        
        # ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
        # bertopicModel = BERTopic(min_topic_size=140, ctfidf_model=ctfidf_model)
        
        
                                             
        #default
        bertopicModel = BERTopic(min_topic_size=min_topic_size)#min_topic_size= 40)#nr_topics=min_topic_size)
        
        
        
        
        
        
        #fit bertopic model to embeddings
        bertopicModel.fit(documents=generalDataset, embeddings=generalEmbeddings)
        
        
        
        
        
        
        
        
        
        ######################### ############## NEW COMMIT
        TD = topicDiversity(bertopicModel.get_topic_info()["Representation"].tolist()) #you give this bertopicmodel.get_topic_info()["Representation"].tolist()
        print("TD: ", TD)
        coherenceTuple = NPMICoherence(bertopicModel, corpusAndDictionaryLabelInc[0], corpusAndDictionaryLabelInc[1])
        print("Coherence: ", coherenceTuple)
        ################


        #THE THEME CONTENT/SPREAD ANALYSIS#################################################
        # #TrainValTest is the training data with its labels #SHOULD BE 0 FOR THE WHOLE TRAINING daTA
        crosstab = compareTrainTopicsToBTopics(bertopicModel, TrainValTest[0], thisModelTrainingEmbeddings)
        

        print("\n\n\n\n\n\n\n\n\n\n\n\n\n\nCROSSTAB: ")
        print(crosstab)
        print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
        crosstab.to_csv("mostRecentCrossTab.csv", index=False)
        #from crosstab or crosstab formatted
        #crosstab.iloc[:, 5]#.sum() #for getting all of column 5 (0 indexed obvs) (and can sum if needed)
        print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
        minusOneTopic = crosstab.iloc[:, 0]#.sum() #for getting all of column 0 (0 indexed obvs) (and can sum if needed)
        print(minusOneTopic.sum())
        #THIS doesnt work due to changing topic reps each iter, print(crosstab["-1_said_woman_people_new"].sum()) #for getting all of column by name
        print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
        print(minusOneTopic.sum()/len(TrainValTest[0]))
        print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
        # crosstab.iloc[2] #for getting all of row 2 (0 indexed)
        # crosstab.loc["Economic/business"] #for getting row by  theme name

        
        # print("Many crosstab stats may be unnecessary in the final version")
        # statsFromCT = statsFromCrosstab(crosstab)
        # print("Many crosstab stats may be unnecessary in the final version, but it might not matter")
        # most_common_predictions, prediction_frequency, total_samples_per_prediction, prediction_composition, average_category_spread, category_spread, least_spread_categories, most_spread_categories, BTTrainComp = statsFromCT
        # topicsToThemesDict = topicsToThemes(BTTrainComp)
    
        #here we get predictions for the theme training data (CHECK held out or main?)

        #then we convert to pred by name

        #then we use topics to themes Dict to get topic prediction by theme 

        #then we get the samples true theme label data (cos its the labelled training data)
        #we can use this for accuacy, precision, recall, f1 etc (that might not be useful.)


        #THEN CHECK WITHIN MAYBE VER1ALTERBERTMODEL OR ALTERBERT MODEL OR SOMETING
        #i think thats where the next step of my experiment is?
        #OR IT MIGHT HAVE BEEN THE CROSSTAB ITSELF WITH DIFFERENT PROCESSING AFTER?

        #either way you get information about the theme spread or topic composition etc
        #and use this as basis for further training

        #e.g. doing a sub bertopic model on just the docs from one big condenced topic and training simcse 
        #to split it up more based on its (sub) topics
        #with the hope that that makes it more nuanced back in the main view/splits it up
    
        # END OF THE CALC FOR THE THEME CONTENT/SPREAD ANALYSIS
        ##########################################################################################################
        
        print(bertopicModel.get_topic_info()["Representation"].tolist())
        print(len(bertopicModel.get_topic_info()["Representation"].tolist()))
        print("gendatasetLen: ", len(generalDataset), " generalEmbeddings len: ", len(generalEmbeddings), "\n\n\n\n\n\n")
        
        
        
        print("STORE RESULTS NOW")
        newRow = {"iteration": iteration, "TD": TD, "Coherence": coherenceTuple, "topicSize": min_topic_size, "percTrainInMinusOne": (minusOneTopic.sum()/len(TrainValTest[0]))*100, "numTopicsGenerated": len(bertopicModel.get_topics())}
        newRow = pd.DataFrame([newRow])
        bertResults = pd.concat([bertResults, newRow], axis=0, ignore_index=True)
        #bertResults = bertResults.append({"iteration": iteration, "TD": TD, "Coherence": coherenceTuple})
        
#evaluate and store evaluation OUTSIDE OF LOOP!
print("NOW DONE WITH BERT ITERS")
bertResults.to_csv("bertResults.csv", index=False)
   
