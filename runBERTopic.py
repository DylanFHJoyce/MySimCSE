print("MADE IT TO runBERTopic")

from bertopic import BERTopic
import pickle


#take general embeddings, parameters
bertopicModel = BERTopic()
#fit bertopic model to embeddings

with open("genDatasetProcessed.pkl", "rb") as f:
  generalDataset = pickle.load(f)

with open("thisModelGeneralEmbeddings.pkl", "rb") as f:
  generalEmbeddings = pickle.load(f)

print("gendatasetLen: ", len(generalDataset), " generalEmbeddings len: ", len(generalEmbeddings))

#evaluate and store evaluation
