import subprocess

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
  pass


def runBert():
    command = "conda run -n berTopicEnv python testBert.py"

    subprocess.run(command, shell=True)





#placeholders 
trainingTripletsCSV = "trainingTriplets4000Manual.csv"
learning_rate = 0.1 #5e-1
num_epochs = 3
datasetName = "placeholder"
#for each learning rate etc

runSim(trainingTripletsCSV, learning_rate, num_epochs)

makeEmbeddings()

runBert()
