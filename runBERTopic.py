print("MADE IT TO runBERTopic")

from bertopic import BERTopic
import pickle
import numpy as np




import gensim
from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

print("first need to calculate corpus")
def NPMICoherence(topicModel, tokenized_corpus, corpusDictionary):
    
    # Calculate NPMI coherence
    coherence_model_npmi = gensim.models.CoherenceModel(topics=topicModel.get_topic_info()["Representation"].tolist(), texts=tokenized_corpus, dictionary=corpusDictionary, coherence='c_npmi')
    
    coherence_npmi = coherence_model_npmi.get_coherence()
    
    coherence_npmiPerTopic = coherence_model_npmi.get_coherence_per_topic()

    print("NPMI Coherence Score:", coherence_npmi)

    return coherence_npmi, coherence_npmiPerTopic
    
def topicDiversity(topicsNWordsList):
    
    uniqueSet = set()
    for topicNWords in topicsNWordsList:
        
        thisTopicSet = set(topicNWords)
        uniqueSet = uniqueSet.union(thisTopicSet)
    
    return len(uniqueSet) / (10 *len(topicsNWordsList))










#take general embeddings, parameters


with open("genDatasetProcessed.pkl", "rb") as f:
  generalDataset = pickle.load(f)

with open("thisModelGeneralEmbeddings.pkl", "rb") as f:
  generalEmbeddings = pickle.load(f)




#ALSO LOAD THE EMBEDDING MODEL #DONT NEED TO BECAUSE EMBEDDINGS ARE PRECOMPUTED?
generalEmbeddings = np.load("thisModelGeneralEmbeddings.pkl", allow_pickle=True)


#and provide params here
bertopicModel = BERTopic()
bertopicModel.fit(documents=generalDataset, embeddings=generalEmbeddings)

#fit bertopic model to embeddings


print("gendatasetLen: ", len(generalDataset), " generalEmbeddings len: ", len(generalEmbeddings))

#evaluate and store evaluation
