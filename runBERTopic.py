print("MADE IT TO runBERTopic")

from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
import pickle
import numpy as np




import gensim
from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
import nltk
from nltk.tokenize import word_tokenize

#nltk.download('punkt')

print("first need to calculate corpus")
def NPMICoherence(topicModel, tokenized_corpus, corpusDictionary):
    
    # Calculate NPMI coherence
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



#ALSO LOAD THE EMBEDDING MODEL #DONT NEED TO BECAUSE EMBEDDINGS ARE PRECOMPUTED?
generalEmbeddings = np.load("thisModelGeneralEmbeddings.pkl", allow_pickle=True)
#and provide params here




#with additional params
# ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
# bertopicModel = BERTopic(min_topic_size=140, ctfidf_model=ctfidf_model)


                                     

#default
bertopicModel = BERTopic(min_topic_size=140)






#fit bertopic model to embeddings
bertopicModel.fit(documents=generalDataset, embeddings=generalEmbeddings)









######################### ############## NEW COMMIT
TD = topicDiversity(bertopicModel.get_topic_info()["Representation"].tolist()) #you give this bertopicmodel.get_topic_info()["Representation"].tolist()
print("TD: ", TD)
coherenceTuple = NPMICoherence(bertopicModel, corpusAndDictionaryLabelInc[0], corpusAndDictionaryLabelInc[1])
print("Coherence: ", coherenceTuple)
################

print(bertopicModel.get_topic_info()["Representation"].tolist())
print(len(bertopicModel.get_topic_info()["Representation"].tolist()))


print("gendatasetLen: ", len(generalDataset), " generalEmbeddings len: ", len(generalEmbeddings), "\n\n\n\n\n\n")

#evaluate and store evaluation
