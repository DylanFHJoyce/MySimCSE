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
    # topicReps = topicModel.get_topic_info()[1:]["Representation"].tolist()
    # # Calculate NPMI coherence
    # coherence_model_npmi = gensim.models.CoherenceModel(topics=topicReps, texts=tokenized_corpus, dictionary=corpusDictionary, coherence='c_npmi')
    # # Calculate NPMI coherence
    coherence_model_npmi = gensim.models.CoherenceModel(topics=topicModel.get_topic_info()["Representation"].tolist(), texts=tokenized_corpus, dictionary=corpusDictionary, coherence='c_npmi')
    
    coherence_npmi = coherence_model_npmi.get_coherence()
    
    coherence_npmiPerTopic = coherence_model_npmi.get_coherence_per_topic()
    
    print("NPMI Coherence Score:", coherence_npmi)
    
    return coherence_npmi, coherence_npmiPerTopic
    
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

bertResults = pd.DataFrame(columns=["iteration", "TD", "Coherence"])
bertResults.to_csv("bertResults.csv", index=False)

bertResults = pd.read_csv("bertResults.csv")

#topicSizes = [40]
for iteration in range(0, 1):
    #for min_topic_size in 
    
    # ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
    # bertopicModel = BERTopic(min_topic_size=140, ctfidf_model=ctfidf_model)
    
    
                                         
    #default
    bertopicModel = BERTopic(min_topic_size=50)
    
    
    
    
    
    
    #fit bertopic model to embeddings
    bertopicModel.fit(documents=generalDataset, embeddings=generalEmbeddings)
    
    
    
    
    
    
    
    
    
    ######################### ############## NEW COMMIT
    TD = topicDiversity(bertopicModel.get_topic_info()["Representation"].tolist()) #you give this bertopicmodel.get_topic_info()["Representation"].tolist()
    print("TD: ", TD)
    coherenceTuple = NPMICoherence(bertopicModel, corpusAndDictionaryLabelInc[0], corpusAndDictionaryLabelInc[1])
    print("Coherence: ", coherenceTuple)
    ################

    #TrainValTest is the training data with its labels #SHOULD BE 0 FOR THE WHOLE TRAINING daTA
    crosstab = compareTrainTopicsToBTopics(bertopicModel, TrainValTest[0], thisModelTrainingEmbeddings) 
    print("Many crosstab stats may be unnecessary in the final version")
    statsFromCT = statsFromCrosstab(crosstab)
    print("Many crosstab stats may be unnecessary in the final version, but it might not matter")
    most_common_predictions, prediction_frequency, total_samples_per_prediction, prediction_composition, average_category_spread, category_spread, least_spread_categories, most_spread_categories, BTTrainComp = statsFromCT
    topicsToThemesDict = topicsToThemes(BTTrainComp)




    
    print(bertopicModel.get_topic_info()["Representation"].tolist())
    print(len(bertopicModel.get_topic_info()["Representation"].tolist()))
    print("gendatasetLen: ", len(generalDataset), " generalEmbeddings len: ", len(generalEmbeddings), "\n\n\n\n\n\n")
    
    
    
    print("STORE RESULTS NOW")
    newRow = {"iteration": iteration, "TD": TD, "Coherence": coherenceTuple}
    newRow = pd.DataFrame([newRow])
    bertResults = pd.concat([bertResults, newRow], axis=0, ignore_index=True)
    #bertResults = bertResults.append({"iteration": iteration, "TD": TD, "Coherence": coherenceTuple})
    
#evaluate and store evaluation OUTSIDE OF LOOP!
print("NOW DONE WITH BERT ITERS")
bertResults.to_csv("bertResults.csv", index=False)
   
