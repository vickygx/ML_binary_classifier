# File used to label the data for the challenge
import numpy as np
import project1_code as p1

output_file = "challenge_predictions.txt"
name = "Victoria Gong"
pseudonym = "Vicky"

# Additional features
difficult_words = p1.extract_file_words('SAT_words.txt')
words = p1.extract_file_words('words.txt')
k = 6

train_labels, train_texts = p1.read_data('train.tsv')
dictionary = p1.extract_dictionary(train_texts)
train_feature_matrix = p1.extract_feature_vectors(train_texts, dictionary, difficult_words, words, k)

# Best Lambda, T
T = 20
l = 100

print "Calculating model..."
theta, theta_0 = p1.avg_passive_aggressive(train_feature_matrix, train_labels, T, l)

dummy_labels, test_texts = p1.read_data('submit.tsv')
test_feature_matrix = p1.extract_feature_vectors(test_texts, dictionary, difficult_words, words, k)

print "Making predictions..."
predictions = p1.classify(test_feature_matrix, theta_0, theta)

print "Writing labels to output file: "+str(output_file)
p1.write_submit_predictions(predictions,output_file,name,pseudonym)