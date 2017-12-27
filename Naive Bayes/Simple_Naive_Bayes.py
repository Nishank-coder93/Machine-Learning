# Author: Nishank Bhatnagar
# Machine Learning: Implements Naive Bayes Algortihm for text classification 
# Uses bag of words Technique to calculate the probabilities of the words occuring the document vocabulary
# Evaluates the algorithm using 20Newsgroup data

# Import the necessary modules
import os
import time
import math
import nltk as nl
import numpy as np
import csv

class NaiveBayes():
    def __init__(self):
        # Loads the labels of the categories
        self.categories = os.listdir('20_newsgroups')
        self.file_dict_cat = {}  # Record of all the file paths to be fed in during the pre processing step
        self.prior_prob_dict = {}  # Dictionary for Prior Probability of Docs in each class
        self.cat_common_freq = {}  # Record of occurrence of all words in sample docs for each Class
        self.class_words_prob = {}  # Record for conditional probability of each word in each Class

    """
            This function calculates the prior probability which is the number of probability
            of # of docs in class C by total # of docs 
    """
    def prior_probability(self,newsgroup_num_doc, total_docs):
        for cat, docnum in newsgroup_num_doc.items():
            self.prior_prob_dict[cat] = docnum / total_docs

        print("\n\n")
        csv_open = csv.writer(open('prior_probability.csv', 'w'))
        csv_open.writerow(['Class', 'Probability of Docs for Class'])

        print("================================================================================================")
        print("The Prior probabilities are :")
        for key,val in self.prior_prob_dict.items():
            print(key,":",val)
            csv_open.writerow([key, val])
        print("================================================================================================")
        print("Stored in a file named 'prior_probability.csv' ")
        print("================================================================================================\n\n")



    """
            This function is used to load all the files path data into a dictionary and calculates the 
            prior probability of number of docs 
    """
    def load_files_list_data(self):
        total_docs_num = 0
        newsgroup_num_of_docs = {}  # Record for number of docs in each class

        # Keep record of Files in each Class folder
        for i in range(len(self.categories)):
            self.file_dict_cat[self.categories[i]] = os.listdir('20_newsgroups/' + self.categories[i])

        # Calculates the number of document in each Class
        for cats in self.categories:
            temp_len = len(self.file_dict_cat[cats])
            newsgroup_num_of_docs[cats] = temp_len
            total_docs_num += temp_len

        print("\nThe number of docs present in each Class or Category : ")
        # Calculate the prior probability
        for key, val in newsgroup_num_of_docs.items():
            print(key,":",val)
        self.prior_probability(newsgroup_num_of_docs, total_docs_num)

    """
        This function retrieves the File from given category and filename and returns the Text of the file
    """
    def file_retrieve(self,category, file_name):
        file_path = '20_newsgroups/' + category + '/' + file_name
        fd_one_file = open(file_path, 'rb')
        text = fd_one_file.read()
        fd_one_file.close()
        return str(text.decode("utf-8", 'ignore'))

    """
        This function breaks down the text into sentences and returns the Sentences excluding the beginning section
    """
    def get_sentence(self,text):
        split_text = text.split("\n\n")
        sen = "".join(split_text[1:])
        return sen

    """
        Filters the text and breaks it down into tokens or words 
        This function will also remove all the stop words which are considered the most common words in English Lang
        Returns a list of tokens 
    """
    def filter_text(self,txt):
        tokens = nl.word_tokenize(txt)
        # wrds = [word.lower() for word in tokens if word.isalnum()]
        wrds = [word for word in tokens if word.isalnum()]
        stop_words = set(nl.corpus.stopwords.words('english'))
        wrds = [w.lower() for w in wrds if not w in stop_words]
        return wrds

    """
        Function to count total occurrence of Token/Word in one Class Sample
    """
    def count_occurance(self,hist_word_list):
        hist_freq = {}
        for word_list in hist_word_list:
            for word in word_list:
                hist_freq[word] = hist_freq.get(word, 0) + 1
        return hist_freq

    #####################################################################################
    #                           Pre Processing Step                                     #
    ####################################################################################

    # This function calculates the frequency of Tokens in each file according to each Class
    def calculate_word_app(self,files_name_dict):
        cat_word_dict = {}
        print("\n\n")
        print("================================================================================================")
        print("Loading Files and pre processing..... (Wait time: 1min - 2min (approx))")
        for cat in self.categories:
            # Gets the file path information for all docs in Class C
            files_list = files_name_dict[cat]

            collective_words = []

            # In this step for each file it retrieves the text from the file
            # breaks down the text into sentences and selects the sentences containing actual message or news
            # then breaks down the sentence into Tokens and removes the Stop words
            # then stores the Token list for each document according to each class in a dictionary
            for file_name in files_list:
                f_txt = self.file_retrieve(cat, file_name)
                sent = self.get_sentence(f_txt)
                filtered_text = self.filter_text(sent)
                collective_words.append(filtered_text)
            cat_word_dict[cat] = collective_words
            print("=", end='')

        print("=> 100%")
        print("================================================================================================\n\n")

        return cat_word_dict

    """
        Calculates the Occurrence of all the words in the Sample Docs from each Class
    """
    def calculate_words_freq(self,class_words_dict):
        for cat in self.categories:
            self.cat_common_freq[cat] = self.count_occurance(class_words_dict[cat])

    """
        Calculates the total number of words or vocabulary used for the training sample
    """
    def calculate_total_words(self):
        num_of_words_per_doc = 0
        num_of_total_words = {}

        for key, value in self.cat_common_freq.items():
            for words, freq in value.items():
                num_of_words_per_doc += freq
            num_of_total_words[key] = num_of_words_per_doc
            num_of_words_per_doc = 0

        return num_of_total_words

    """
        Calculates the conditional probability part which nothing but the Probability of each Token present 
        in each class by total # of words or vocab used for the training sample 
    """
    def conditional_probability(self):
        num_of_tot_words = self.calculate_total_words()
        for cat, words in self.cat_common_freq.items():
            temp_cat_words_prob = {}
            for word, freq in words.items():
                temp_cat_words_prob[word] = freq / num_of_tot_words[cat]

            self.class_words_prob[cat] = temp_cat_words_prob

    def write_to_file(self):
        print("================================================================================================")
        print(" The words probability is loaded in a file named 'word_prob_file.csv ")
        print("================================================================================================\n\n")
        word_prob_file = csv.writer(open('word_prob_file.csv', 'w'))
        word_prob_file.writerow(['Category/Class', 'Words/Tokens', 'Probability'])

        for cat, values in self.class_words_prob.items():
            word_prob_file.writerow([cat])

            for word, prob in values.items():
                word_prob_file.writerow(['',word,prob])

    #####################################################################################
    #             Training Step begins here and helper methods are above                #
    ####################################################################################

    def training_step(self):
        # This Loads all the file names and stores it in a dictionary which is preprocessed later
        # and calculates the prior probabilities
        self.load_files_list_data()

        print("\n\n")
        print("================================================================================================")
        print("The categories found are : ")
        print(self.categories)
        print("================================================================================================\n\n")

        # Calculates the time it takes to pre process the text files that are loaded
        start_time = time.time()

        print("Loading training data please wait ....")
        # This is the main point where the pre processing takes place
        cat_word_dictionary = self.calculate_word_app(self.file_dict_cat)

        end_time = time.time()
        total_time = end_time - start_time
        print(total_time / 60, " Minutes to load")

        # Calculates the word frequency for each document Class
        self.calculate_words_freq(cat_word_dictionary)

        # Calculates the probability of each token or word present in all sample documents of a class
        self.conditional_probability()

        self.write_to_file()

        print("\n\n")
        print("================================================================================================")
        print("Model is completely trained on Sample Training Set ")
        print("================================================================================================\n\n")


    #####################################################################################
    #                    Start of Predicting the class of sample data                   #
    ####################################################################################

    def load_sample_data(self):
        print("Loading Sample data ... ")
        # Load Sample Documents
        sample_file_dict = {}

        # Keep record of Files in each Class folder
        for i in range(len(self.categories)):
            sample_file_dict[self.categories[i]] = os.listdir('mini_newsgroups/' + self.categories[i])

        return self.calculate_word_app(sample_file_dict)

    """
        This function gets the probability of each word in the document: 
        If the word is not present then the probability is not considered --> 
        since we are not multiplying the probabilities and calculating the sum of logs 
        we can leave the words whose probabilities are not calculated for the Class C 
    """
    def predict_class(self,document):
        max_prob = 0
        predicted_category = ""
        for cat, cat_prob in self.prior_prob_dict.items():
            total_prob = abs(math.log(cat_prob))
            for word in document:
                prob_value = self.class_words_prob[cat].get(word, 0)
                if prob_value != 0:
                    total_prob += abs(math.log(prob_value))
                else:
                    total_prob += prob_value


            if total_prob > max_prob:
                max_prob = total_prob
                predicted_category = cat

        return predicted_category

    """
        This function is used to Predict the Class of the Testing documents 
    """
    def predicting(self):
        # Sample or Test data is loaded
        sample_word_dict = self.load_sample_data()
        actual_predicted_match_vals = []
        predicted_classes = []
        actual_classes = []

        # gets the predicted Class or Category of the document and appedns into a list
        for cat, doc_list in sample_word_dict.items():
            for doc in doc_list:
                predicted_class = self.predict_class(doc)
                predicted_classes.append(predicted_class)

                actual_classes.append(cat)

                if predicted_class == cat:
                    actual_predicted_match_vals.append(0)
                else:
                    actual_predicted_match_vals.append(1)

        csv_open = csv.writer(open('actual_predicted_values.csv', 'w'))
        csv_open.writerow(['Predicted Values','Actual Values'])

        for i in range(len(predicted_classes)):
            csv_open.writerow([predicted_classes[i], actual_classes[i]])

        print("\n\n")
        print("================================================================================================")
        print('The Predicted Values and Actual Values are stored in file named "actual_predicted_values.csv"')
        print("================================================================================================\n\n")

        # print("Classes Predicted from Sample or Test data set")
        # print(predicted_classes)
        # print("The Actual Classes ")
        # print(actual_classes)
        return np.array(actual_predicted_match_vals)

    def accuracy(self):
        actual_predicted_match_vals = self.predicting()
        # print(actual_predicted_match_vals)
        cnt_ones = np.count_nonzero(np.array(actual_predicted_match_vals))
        cnt_zeroes = 2000 - cnt_ones
        print("================================================================================================")
        print("Accuracy: ", (cnt_zeroes / 2000) * 100, "%")
        print("================================================================================================")

if __name__ == '__main__':

    nb = NaiveBayes()

    # Calls the training step to train the system on Training sample document
    nb.training_step()

    # Predicts the probability of Class for each document and calculates the accuracy by comparing with actual values
    nb.accuracy()
