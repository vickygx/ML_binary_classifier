# Script skeleton to call functions from project1_code and
# to run experiments
import numpy as np
import project1_code as p1

#########################################################################
########################  PART 1: TOY EXAMPLE ###########################
#########################################################################
'''
"""
TODO: 
    Implement the following functions in project1_code.py:
         1) perceptron 
         2) avg_perceptron 
         3) avg_passive_agressive 
"""

# Read data
toy_labels, toy_data = p1.read_toy_data('toy_data.tsv')

# Train classifiers
T = 5
l = 10.0   
   
theta, theta_0 = p1.perceptron(toy_data, toy_labels, T)
p1.plot_2d_examples(toy_data, toy_labels, theta_0, theta, 'Perceptron')
theta, theta_0 = p1.avg_perceptron(toy_data, toy_labels, T)
p1.plot_2d_examples(toy_data, toy_labels, theta_0, theta, 'Averaged Perceptron')
theta, theta_0 = p1.avg_passive_aggressive(toy_data, toy_labels, T, l)
p1.plot_2d_examples(toy_data, toy_labels, theta_0, theta, 'Passive-Agressive')

'''



##########################################################################
######################## PART 2 : ESSAY DATA #############################
#########################################################################


########## READ DATA ##########
# Additional features
difficult_words = p1.extract_file_words('SAT_words.txt')
words = p1.extract_file_words('words.txt')
k = 6

#Training data
train_labels, train_text = p1.read_data('train.tsv')
dictionary = p1.extract_dictionary(train_text)
train_feature_matrix = p1.extract_feature_vectors(train_text, dictionary, difficult_words, words, k)

#Validation data
val_labels, val_text = p1.read_data('validation.tsv')
val_feature_matrix = p1.extract_feature_vectors(val_text, dictionary, difficult_words, words, k)

#Test data
test_labels, test_text = p1.read_data('test.tsv')
test_feature_matrix = p1.extract_feature_vectors(test_text, dictionary, difficult_words, words, k)


########### EVALUATE PERFORMANCE ON TEST SET #############
'''
T = 5
l = 1.0

# Performance of perceptron
theta, theta_0 = p1.perceptron(train_feature_matrix, train_labels, T)
train_label_vector= p1.classify(train_feature_matrix, theta_0, theta)
test_label_vector= p1.classify(test_feature_matrix, theta_0, theta)

print "perceptron train", p1.score_accuracy(train_label_vector, train_labels)
print "perceptron test", p1.score_accuracy(test_label_vector, test_labels)

# Performance of avg. perceptron
theta, theta_0 = p1.avg_perceptron(train_feature_matrix, train_labels, T)
train_label_vector= p1.classify(train_feature_matrix, theta_0, theta)
test_label_vector= p1.classify(test_feature_matrix, theta_0, theta)

print "avg. perceptron train", p1.score_accuracy(train_label_vector, train_labels)
print "avg. perceptron test", p1.score_accuracy(test_label_vector, test_labels)

# Performance of avg. passive aggressive
theta, theta_0 = p1.avg_passive_aggressive(train_feature_matrix, train_labels, T, l)
train_label_vector= p1.classify(train_feature_matrix, theta_0, theta)
test_label_vector= p1.classify(test_feature_matrix, theta_0, theta)

print "avg. passive aggressive train", p1.score_accuracy(train_label_vector, train_labels)
print "avg. passive aggressive test", p1.score_accuracy(test_label_vector, test_labels)

'''

########## TRAINING + VALIDATION TO TUNE T AND L ##########

"""
1) For multiple values of T and lambda:
    - Predict labels for validation set (using function 'classify')
    - Calculate validation accuracy (using function 'score_accuracy')
2) Choose optimal learning method and parameters based on validation accuracy
"""

T = 1
l_values = [1.0, 10.0, 20.0, 50.0, 100.0]
for l in l_values:
    print "T:", T, "  l:", l 

    print "avg passive agressive:"
    theta, theta_0 = p1.avg_passive_aggressive(train_feature_matrix, train_labels, T, l)
    train_label_vector = p1.classify(train_feature_matrix, theta_0, theta)
    val_label_vector = p1.classify(val_feature_matrix, theta_0, theta)
    
    print "train score accuracy", p1.score_accuracy(train_label_vector, train_labels)
    print "val score accuracy", p1.score_accuracy(val_label_vector, val_labels)


########## PLOTTING ACCURACY WITH DIFFERENT T AND L ##########
'''
# Passive aggressive lambda when T = 20
pa_lambda_param_values = [1, 10, 20, 50, 100]
pa_lambda_train_scores = [93.50, 93.61, 93.67, 93.61, 93.50]
pa_lambda_validation_scores = [86.86, 87.43, 87.43, 88.0, 88.0]

p1.plot_scores('Lambda', pa_lambda_param_values, pa_lambda_train_scores, pa_lambda_validation_scores, 'Passive Aggressive T = 20')


# Passive aggressive T when lambda = 50
pa_T_param_values = [1, 5, 10, 15, 20]
pa_T_train_scores = [77.11, 87.89, 91.06, 92.83, 93.61]
pa_T_validation_scores = [78.57, 85.14, 86.57, 87.43, 88]

p1.plot_scores('T', pa_T_param_values, pa_T_train_scores, pa_T_validation_scores, 'Passive Aggressive L = 50')

# Perceptron T
p_T_param_values = [1, 5, 10, 15, 20]
p_T_train_scores = [75.5, 87.22, 91.0, 92.83, 93.94]
p_T_validation_scores = [76.57, 84.28, 86.85, 87.14, 87.14]

p1.plot_scores('T', p_T_param_values, p_T_train_scores, p_T_validation_scores, 'Perceptron')

'''

########## TESTING OPTIMAL PARAMETERS ##########
'''
T_optimal = 20
lambda_optimal = 50

optimal_theta, optimal_theta_0 = p1.avg_passive_aggressive(train_feature_matrix, train_labels, T_optimal, lambda_optimal)
predictions = p1.classify(test_feature_matrix, optimal_theta_0, optimal_theta)

print 'Performance on Test Set:'
test_score = p1.score_accuracy(predictions, test_labels)
'''
