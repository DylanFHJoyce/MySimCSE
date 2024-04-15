print("MADE IT TO runBERTopic")

from bertopic import BERTopic
import pickle
import numpy as np

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
