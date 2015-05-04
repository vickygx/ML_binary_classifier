# File containing useful functions and 
# function templates to implement learning algorithms
from string import punctuation, digits
import numpy as np
import matplotlib.pyplot as plt
import re

text_length_quantiles = 0
word_length_quantiles = 0

def read_data(filepath):
    """
    Returns an array of labels and an array of the corresponding texts
    """
    f = open(filepath, 'r')
    all_labels = []
    all_texts = []
    for line in f:
        label, text = line.split('\t')
        all_labels.append(int(label))
        all_texts.append(text)
    return (all_labels, all_texts)

def read_toy_data(filepath):
    """
    Returns (labels, data) for toy data
    """
    f = open(filepath, 'r')
    toy_labels = []
    toy_data = []
    for line in f:
        label, x, y = line.split('\t')
        toy_labels.append(int(label))
        toy_data.append([float(x), float(y)])
    return (toy_labels, np.array(toy_data))

def extract_words(input_string):
    """
      Returns a list of lowercase words in a string.
      Punctuation and digits are separated out into their own words.
    """
    for c in punctuation + digits:
        input_string = input_string.replace(c, ' ' + c + ' ')
    
    return input_string.lower().split()

def extract_dictionary(texts):
    """
      Given an array of texts, returns a dictionary of unigrams and bigrams.
      Each line is passed into extract_words, and a list on unique
      unigrams and bigrams is maintained.
      
      In addition, it computes quartiles of text length (in # of words) and word
      length (in # of characters).
    """
    global text_length_quantiles
    global word_length_quantiles
    unigrams = set([])
    bigrams  = set([])
    text_lengths = []
    word_lengths = []
    for text in texts:
        word_list = extract_words(text)
        text_lengths.append(len(word_list))
        for i, word in enumerate(word_list):
            word_lengths.append(len(word))
            if(word not in unigrams):
                unigrams.add(word)
            if(i > 0):
                bigram = previous_word + '_' + word
                if(bigram not in bigrams):
                    bigrams.add(bigram)
            previous_word = word
    dictionary = unigrams.union(bigrams)
    text_length_quantiles = [np.percentile(text_lengths, k) for k in [25,50,75]]
    word_length_quantiles = [np.percentile(word_lengths, k) for k in [25,50,75]]
    return {word:i for i, word in enumerate(dictionary)}


def extract_file_words(filepath):
    """
    Returns an dictionary of (word, 1) for each 
    word in a file where each word is deliminated by a new line
    """

    with open(filepath, 'r') as f:
        content = f.read()
    words = content.rstrip().split('\n') 
        
    return dict([(word, 1) for word in words]) 

def extract_feature_vectors(texts, dictionary, difficult_words, words, k):
    """
      Returns the feature representation of the data.
      The returned matrix is of shape (n, m), where n is the number of texts
      and m the total number of features (entries in the dictionary and any 
      additional feature).
    """
    
    num_of_new_features = 8 # Number of features other than bag of words
    num_texts = len(texts)
    feature_matrix = np.zeros([num_texts, len(dictionary) + num_of_new_features])


    for i, text in enumerate(texts):
        num_difficult_words = 0
        num_words_greater_k = 0
        num_misspelt_words = 0
        #### Unigrams and bigrams
        word_list = extract_words(text)
        num_words = len(word_list)
        for j,word in enumerate(word_list):
            if(word in dictionary):
                feature_matrix[i, dictionary[word]] = 1
            # Occurence of misspelt words
            elif(word not in words):
                num_misspelt_words += 1
            if(j > 0):
                bigram = previous_word + '_' + word
                if(bigram in dictionary):
                    feature_matrix[i, dictionary[bigram]] = 1
            # Counting occurence of "difficult" words
            if(word in difficult_words):
                num_difficult_words += 1
            if(len(word) > k):
                num_words_greater_k += 1
            previous_word = word
        
        #### Additional Features
        # Binary features for text length
        feature_matrix[i, len(dictionary) + 0] = (num_words < text_length_quantiles[0]) # Bottom 25% 
        feature_matrix[i, len(dictionary) + 1] = (num_words < text_length_quantiles[1]) # Bottom 50% 
        feature_matrix[i, len(dictionary) + 2] = (num_words < text_length_quantiles[2]) # Bottom 75%
        lengths = [len(w) for w in word_list]
        feature_matrix[i, len(dictionary) + 3] = sum(lengths)/float(len(lengths)) # Average word length
        feature_matrix[i, len(dictionary) + 4] = len(set(word_list)) # Unique words
        # Number of difficult words
        feature_matrix[i, len(dictionary) + 5] = num_difficult_words
        # Number of words greater than k
        feature_matrix[i, len(dictionary) + 6] = num_words_greater_k
        # Number of misspelt words
        feature_matrix[i, len(dictionary) + 7] = num_misspelt_words

    return feature_matrix

def perceptron(feature_matrix, labels, T):
    theta = len(feature_matrix[0]) * [0.0]
    theta_0 = 0.0

    # def h(x, theta, theta_0):
    #     if np.dot(x, theta) + theta_0 <= 0:
    #         return -1
    #     else: 
    #         return 1

    for _ in range(T):
        for i in range(len(feature_matrix)):
            # if (h(feature_matrix[i], theta, theta_0) != labels[i]):
            if ((np.dot(theta,feature_matrix[i]) + theta_0 ) * labels[i] <= 0):
                theta = np.add(theta, np.multiply(labels[i], feature_matrix[i]))
                theta_0 = theta_0 + labels[i]
 
   
    # print "perceptron values:", theta, theta_0
    return [theta, theta_0]

def avg_perceptron(feature_matrix, labels, T):
    thetas = []
    thetas_0 = []
    theta = len(feature_matrix[0]) * [0.0]
    theta_0 = 0.0

    for _ in range(T):
        for i in range(len(feature_matrix)):
            if ((np.dot(theta,feature_matrix[i]) + theta_0 ) * labels[i] <= 0):
                theta = np.add(theta, np.multiply(labels[i], feature_matrix[i]))
                theta_0 = theta_0 + labels[i]
                
            thetas.append(theta)
            thetas_0.append(theta_0)

    nT = float(len(feature_matrix) * T)
    theta_avg = [sum(theta[i] for theta in thetas)/(nT) for i in range(len(feature_matrix[0]))]
    theta_0_avg = sum(theta_0 for theta_0 in thetas_0) / (nT)
    
    # print "avg perceptron values:", theta_avg, theta_0_avg
    return [theta_avg, theta_0_avg]

def avg_passive_aggressive(feature_matrix, labels, T, l):
    thetas = []
    thetas_0 = []
    theta = len(feature_matrix[0]) * [0.0]
    theta_0 = 0.0

    def loss(theta_k, theta_k_0, x,y):
        current = 1.0 - ((np.dot(theta_k,x) + theta_k_0)*y)
        return max(current, 0.0)

    for _ in range(T):
        for i in range(len(feature_matrix)):
            x = feature_matrix[i]
            y = labels[i]
            step = min(loss(theta, theta_0, x,y)/(np.linalg.norm(x)**2.0), 1.0/l)
            
            theta = np.add(theta, np.multiply(labels[i]*step, feature_matrix[i]))
            thetas.append(theta)
            theta_0 = theta_0 + (step * y)
            thetas_0.append(theta_0)

    nT = float(len(feature_matrix) * T)
    theta_avg = [sum(theta[i] for theta in thetas) / nT for i in range(len(feature_matrix[0]))]
    theta_0_avg = sum(theta_0 for theta_0 in thetas_0) / nT
    
    return [theta_avg, theta_0_avg]

def classify(feature_matrix, theta_0, theta_vector):
    """
      TODO: IMPLEMENT FUNCTION
      Classifies a set of data points given a weight vector and offset.
      Inputs are an (m, n) matrix of input vectors (m data points and n features),
      a real number offset, and a length n parameter vector.
      Returns a length m label vector.
    """
    return [(1 if (np.dot(theta_vector, x) + theta_0) > 0 else -1) for x in feature_matrix]
    
def score_accuracy(predictions, true_labels):
    """
    Inputs:
        - predictions: array of length (n,1) containing 1s and -1s
        - true_labels: array of length (n,1) containing 1s and -1s
    Output:
        - percentage of correctly predicted labels
    """
    correct = 0
    for i in xrange(0, len(true_labels)):
        if(predictions[i] == true_labels[i]):
            correct = correct + 1
    
    percentage_correct = 100.0 * correct / len(true_labels)
    print("Method gets " + str(percentage_correct) + "% correct (" + str(correct) + " out of " + str(len(true_labels)) + ").")
    return percentage_correct

    
def write_submit_predictions(labels,outfile,name,pseudonym="Anonymous"):
    """
      Outputs your label predictions to a given file.
      Prints name on the first row.
      labels must be a list of length 500, consisting only of 1 and -1
    """
    
    if(len(labels) != 500):
        print("Error - output vector should have length 500.")
        print("Aborting write.")
        return
    
    with open(outfile,'w') as f:
        f.write("%s\n" % name)
        f.write("%s\n" % pseudonym)
        for value in labels:
            if((value != -1.0) and (value != 1.0)):
                print("Invalid value in input vector.")
                print("Aborting write.")
                return
            else:
                f.write("%i\n" % value)
    print('Completed writing predictions successfully.')



def plot_2d_examples(feature_matrix, labels, theta_0, theta, title):
    """
      Uses Matplotlib to plot a set of labeled instances, and
      a decision boundary line.
      Inputs: an (m, 2) feature_matrix (m data points each with
      2 features), a length-m label vector, and hyper-plane
      parameters theta_0 and length-2 vector theta.
    """
    
    cols = []
    xs = []
    ys = []
    
    for i in xrange(0, len(labels)):
        if(labels[i] == 1):
            cols.append('b')
        else:
            cols.append('r')
        xs.append(feature_matrix[i][0])
        ys.append(feature_matrix[i][1])
    
    plt.scatter(xs, ys, s=40, c=cols)
    [xmin, xmax, ymin, ymax] = plt.axis()
    
    linex = []
    liney = []
    for x in np.linspace(xmin, xmax):
        linex.append(x)
        if(theta[1] != 0.0):
            y = (-theta_0 - theta[0]*x) / (theta[1])
            liney.append(y)
        else:
            liney.append(0)
    plt.suptitle(title, fontsize=15)
    plt.plot(linex, liney, 'k-')
    plt.show()


def plot_scores(parameter,parameter_values,train_scores,validation_scores,title):
    """
      Uses Matplotlib to plot scores as a function of hyperparameters.
      Inputs:
           - parameter:  string, one of 'Lambda' or 'Iterations'
           - parameter_values: a list n of parameter values
           - train_scores: a list of n scores on training data
           - validations:  a list of n scores on validation data
           - title: String
    """
    
    plt.plot(parameter_values,train_scores,'-o')
    plt.plot(parameter_values,validation_scores,'-o')
    plt.legend(['Training Set','Validation Set'], loc='upper right')
    plt.title(title)
    plt.xlabel('Hyperparameter: ' + parameter)
    plt.ylabel('Accuracy (%)')
    plt.show()
