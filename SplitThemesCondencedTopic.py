print("test print")

#have the sections all in here and comment out at whatever stage in the process, its messy but i need to save time.



#we have a model to start with
#we use it to encode our data etc



#we run a few bert models on it with various parameters, recording the themes that are often deemed condenced by our metrics
#(so our normal spread analysis but we count the times a theme is condenced and add it to the overall count)
#we proceede with one or a few of the most condenced themes


#we run a final bert model to generate a midling amount of topics, we use fit transform to create this model so that we have 
#the topic allocations for the samples in the wider dataset

#we choose one of the topics that has a condenced amount of our target theme
#THIS SHOULD BE A TOPIC WITH ABOVE 1000 SAMPLES FROM THE GENERAL DATASET!
#we run a bertopic model on just the data from that topic, again fit transform to get the different groupings
#we then use this data to generate a triplet dataset


we add our new triplet dataset to the old one and train all together (if we cant work out the other training)
