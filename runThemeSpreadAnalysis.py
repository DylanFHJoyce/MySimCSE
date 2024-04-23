print("THEME SPREAD TEST")
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


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix



print("first need to calculate corpus (IF NOT ALREADY DONE EXTERNALLY)")
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

    print("(this note is for if you are continuing devlopment of theme hierarchy: but if the theme contains no samples... for all dict entries that are still empty: start at lowest hierarchy layer?")


    return bertTopicCorrelateDict







print("FOR LATER VERSION CONSIDER USING HIGHER GEN DATASET SAMPLE SIZE AS MAY NEED MORE DENSITY FOR SUBMODELING")
#take general embeddings, parameters

#get the proceessed general dataset list 
with open("genDatasetProcessed.pkl", "rb") as f:
  generalDataset = pickle.load(f)




#get the embeddings to then be theme analysed
with open("ThemeSpreadEmbeddings.pkl", "rb") as f:
  ThemeSpreadEmbeddings = pickle.load(f)
  #print("GENSET TYPE", type(ThemeSpreadEmbeddings))





#open  the laelled data (format train, val, test)
with open('split4000Manual.pkl', 'rb') as f:
    TrainValTest = pickle.load(f)



with open("ThemeFocusedTrainingEmbeddings.pkl", "rb") as f:
    ThemeFocusedTrainingEmbeddings = pickle.load(f)

with open("ThemeFocusedValEmbeddings.pkl", "rb") as f:
    ThemeFocusedValEmbeddings = pickle.load(f)

with open("ThemeFocusedTestEmbeddings.pkl", "rb") as f:
    ThemeFocusedTestEmbeddings = pickle.load(f)



print(len(ThemeSpreadEmbeddings), len(generalDataset))

#do bert model on gen dataset

# #themesList = list(set(TrainValTest[0]["Category"].tolist()))

# ThemeSpreadAnalysisBertResults = pd.DataFrame(columns=["iteration", "TD", "Coherence", "topicSize", "percTrainInMinusOne", "numTopicsGenerated"])
# ThemeSpreadAnalysisBertResults.to_csv("ThemeSpreadAnalysisBertResults.csv", index=False)
# ThemeSpreadAnalysisBertResults = pd.read_csv("ThemeSpreadAnalysisBertResults.csv")



bertopicModel = BERTopic(min_topic_size=40)#min_topic_size=min_topic_size)
bertopicModel.fit(documents=generalDataset, embeddings=ThemeSpreadEmbeddings)
minusOneTopicName = bertopicModel.get_topic_info().iloc[0]["Name"]
print(minusOneTopicName)


print("WE SKIP GENERATIONS WITH VERY LOW TOPIC QUANTITIES, IF IT HAPPENS CONSISTENTLY THEN CHECK PARAMS")
if (len(bertopicModel.get_topics()) < 10):
    print("\n\n\n\nA BERTOPIC GENERATION HAS BEEN SKIPPED IN runThemeSpreadAnalysis.py\n\n\n\n")
else:
        
    crosstab = compareTrainTopicsToBTopics(bertopicModel, TrainValTest[0], ThemeFocusedTrainingEmbeddings)
    #minusOneTopic = crosstab.iloc[:, 0]
    
    
    # #NUMBER OF SAMPLES IN TOP 12345 TOPICS FOR EACH THEME (INCLUDING -1)
    # print("\n\nNUMBER OF SAMPLES IN TOP 12345 TOPICS FOR EACH THEME (INCLUDING -1)")
    # quantInTop12345 = []
    # for idx, row in crosstab.iterrows():
    #     print(idx)
    #     print(row)
    #     print("Quantity in -1: ", row.loc[minusOneTopicName])
    #     SV = sorted(row, reverse=True)
    #     print(SV[:10])
    #     quantInTop12345 = [SV[0], sum(SV[:2]), sum(SV[:3]), sum(SV[:4]), sum(SV[:5])]
    #     total = sum(SV)
    #     print((SV[0]/total), (sum(SV[:2])/total), (sum(SV[:3])/total), (sum(SV[:4])/total), (sum(SV[:5])/total))
    #     print(quantInTop12345, "\n")



   
    for idx, row in crosstab.iterrows():
        print(idx)
        print(row)
        print("Quantity in -1: ", row.loc[minusOneTopicName])
        
        rowNoMinus = row.drop(minusOneTopicName)
        SV = sorted(row, reverse=True)
        SVNoMinus = sorted(rowNoMinus, reverse=True)

        quantInTop12345 = [SV[0], sum(SV[:2]), sum(SV[:3]), sum(SV[:4]), sum(SV[:5])]
        quantInTop12345NoMinus = [SVNoMinus[0], sum(SVNoMinus[:2]), sum(SVNoMinus[:3]), sum(SVNoMinus[:4]), sum(SVNoMinus[:5])]
        
        total = sum(SV)
        totalNoMinus = sum(SVNoMinus)
        
        #print((SV[0]/total), (sum(SV[:2])/total), (sum(SV[:3])/total), (sum(SV[:4])/total), (sum(SV[:5])/total))
        percentages = [(value / total) for value in quantInTop12345]
        percentagesNoMinus = [(value / totalNoMinus) for value in quantInTop12345NoMinus]
        percentagesOutOfFullTotal = [(value / total) for value in quantInTop12345NoMinus]
        print("%: ", percentages)
        print("% NO MINUS: ", percentagesNoMinus)
        print("% no minus but out of full number of samples: ", percentagesOutOfFullTotal)
        print("with minus: ", quantInTop12345, "\n")
        print("no minus: ", quantInTop12345NoMinus, "\n")

        if percentagesOutOfFullTotal[0] < 0.2:
            print("\n\n\nTheme: ", idx, "May be quite spread out or much in -1\n(we should probably use a check on the last of the values being under x instead of the first, both work but for this the last is more clear)\n\n\n\n\n")
        if percentagesOutOfFullTotal[0] > 0.8:
            print("\n\n\nTheme ", idx, "May be quite condenced\n\n\n\n\n")



    condThreshold = 8.5
    mixedThreshold = 0.3
    for column in crosstab.columns:
        colVals = crosstab[column]
        colTotal = colVals.sum()

        percentage = colVals / colTotal
        
        if (percentage > condThreshold).any():
            print(column, percentage)
            print(column, " is condenced (more than 85% composed of a single theme) maybe train to split it if it contains most of the samples for that theme\n\n")
        if not (percentage > mixedThreshold).any():
            print(column, percentage)
            print(column, " is mixed (less than 30% of any single theme) maybe train to split it if its a large topic\n\n")
        
    
    # #NUMBER OF SAMPLES IN TOP 12345 TOPICS FOR EACH THEME (WITHOUT -1)
    # print("\n\nNUMBER OF SAMPLES IN TOP 12345 TOPICS FOR EACH THEME (WITHOUT -1)")
    # print("before training this will probably be much lower than the one including -1")
    # quantInTop12345NoMinus = []
    # crossTabNoMinus = crosstab.iloc[:, 1:]
    # for idx, row in crossTabNoMinus.iterrows():
    #     print(idx)
    #     SV = sorted(row, reverse=True)
    #     print(SV[:10])
    #     quantInTop12345NoMinus = [SV[0], sum(SV[:2]), sum(SV[:3]), sum(SV[:4]), sum(SV[:5])]
    #     total = sum(SV)
    #     print("below doesnt count the -1 so is also inaccurate")
    #     print((SV[0]/total), (sum(SV[:2])/total), (sum(SV[:3])/total), (sum(SV[:4])/total), (sum(SV[:5])/total))
    #     print(quantInTop12345, "\n")
    
    
    
    
    
    
    allRowsTotal = 0
    allMinusOneTotal = 0
    for idx, row in crosstab.iterrows():
        rowsTotal = row.sum()
        allRowsTotal = allRowsTotal + rowsTotal
    
        allMinusOneTotal = allMinusOneTotal + row[0]
        print(idx, (row[0] / rowsTotal) * 100)
    print("average % in minus one: ", (allMinusOneTotal/allRowsTotal) * 100)
    
    print(bertopicModel.get_topic_info())
    
    



    
    #make blank matrix
    print("consider making the df into % first and ignoring any V minor occurances? might not matter if we're just selcting the most anyway?")
    coOccurrenceMatrix = np.zeros((len(crosstab.index), len(crosstab.index)), dtype=int)
    for column in crosstab.columns: #for each topic colum
        topic = crosstab[column] #topic = that columns values 
        # Find the themes present in this topic
        presentThemes = topic.index[topic > 0] #get any themes that occur in this topic

        #this section counts for each theme which other theme occurs and adds it to the matrix
        for i, theme1 in enumerate(presentThemes):
            for j, theme2 in enumerate(presentThemes):
                if i < j:
                    #get name of idx
                    idxTheme1 = crosstab.index.get_loc(theme1)
                    idxTheme2 = crosstab.index.get_loc(theme2)
                    coOccurrenceMatrix[idxTheme1, idxTheme2] += topic.iloc[idxTheme1] * topic.iloc[idxTheme2]
                    coOccurrenceMatrix[idxTheme2, idxTheme1] += topic.iloc[idxTheme1] * topic.iloc[idxTheme2] 

    showN = 6
    
    topIdxs = np.argpartition(-coOccurrenceMatrix.flatten(), showN)[:showN]
    
    for index in top_indices:
        i, j = np.unravel_index(index, coOccurrenceMatrix.shape)
        theme1 = crosstab.index[i]
        theme2 = crosstab.index[j]
        print("Themes ", theme1, " and ", theme2, " co-occur the most in the same topics.")
    
    # # then we take 
    # maxCoOccurrences = np.max(coOccurrenceMatrix)
    # themeIndices = np.where(coOccurrenceMatrix == maxCoOccurrences)
    # # show the themes with the maximum co-occurrences
    # for i, j in zip(themeIndices[0], themeIndices[1]):
    #     theme1 = crosstab.index[i]
    #     theme2 = crosstab.index[j]
    #     print("Themes ", theme1, " and ", theme2, " co-occur the most in the same topics.")











    # statsFromCT = statsFromCrosstab(crosstab)
    # most_common_predictions, prediction_frequency, total_samples_per_prediction, prediction_composition, average_category_spread, category_spread, least_spread_categories, most_spread_categories, BTTrainComp = statsFromCT
    
    # #THIS LEAVES SOME TOPICS WITHOUT A THEME IF THEY DID NOT CONTAIN TRAINING SAMPLES
    # topicsToThemesDict = topicsToThemes(BTTrainComp) #make a dict for which theme each topic belongs to 

    # #could do these metrics by theme
    
    # pred, _ = bertopicModel.transform(TrainValTest[1]["Document"].tolist(), ThemeFocusedValEmbeddings)
    # predByName = convertTopicNumToName(pred, bertopicModel.get_topic_info())
    # predByTheme = [topicsToThemesDict.get(key, "Unclassified") for key in predByName] #unclassified indicated there were no training samples in that topic
    # print("HERE COULD DO COSINE SIM TO ASSIGN REMAINING TOPICS TO THEME BASED ON SIM TO ALREADY ASSIGNED TOPICS")
    
    # themeTrueLabel = TrainValTest[1]["Category"].tolist()
    
    # # Calculate accuracy
    # accuracy = accuracy_score(themeTrueLabel, predByTheme)
    # print("Accuracy:", accuracy)
    
    # # Calculate precision
    # precision = precision_score(themeTrueLabel, predByTheme, average='weighted')
    # print("Precision:", precision)
    
    # # Calculate recall
    # recall = recall_score(themeTrueLabel, predByTheme, average='weighted')
    # print("Recall:", recall)
    
    # # Calculate F1 score
    # f1 = f1_score(themeTrueLabel, predByTheme, average='weighted')
    # print("F1 Score:", f1)
    
    # # # Calculate confusion matrix
    # # conf_matrix = confusion_matrix(themeTrueLabel, predByTheme)
    # # print("Confusion Matrix:\n", conf_matrix)
    
    



# exampleColName = crosstab.columns[1]
# for val in crosstab[exampleColName]:
#     print(val)



print("if doing entropy then need all columns as count, even blank ones?")
    #then i also want the enthropy for each theme with and without the -1 topic
    #and keep in mid the difference that will make

    


#take training data embeddings and transform to the bert model

#do spread analysis of the most 


