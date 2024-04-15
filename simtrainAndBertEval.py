import subprocess




def runSim(learning_rate, num_epochs):
  command = (
    "conda run -n simEnv python train.py "
    "--model_name_or_path princeton-nlp/sup-simcse-bert-base-uncased"
    "--train_file news2234_triplet_dataset.csv"
    "--output_dir aTrainedModel"
    f"--num_train_epochs {num_epochs}"
    "--per_device_train_batch_size 64"
    f"--learning_rate {learning_rate}"
    "--max_seq_length 64"
    "--load_best_model_at_end"
    "--pooler_type cls"
    "--overwrite_output_dir"
    "--temp 0.05"
     "--do_train"
     "--fp16"
     "--use_in_batch_instances_as_negatives"
  )
  subprocess.run(command, shell=True)



def runBert():
    command = "conda run -n berTopicEnv python testBert.py"

    subprocess.run(command, shell=True)



#for each learning rate etc

#placeholder
learning_rate = 0.1 #5e-1
num_epochs = 5
batch_size = 32


runSim(learning_rate, num_epochs)


runBert()
