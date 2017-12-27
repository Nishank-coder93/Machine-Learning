================================================
		RUNNING THE PROGRAM
================================================

- This program will require Python 3.X or higher to run the program
- navigate to the folder where the program resides using CD (Change Directory)
- Then install all the dependent libraries using 

[pip install -r requirements.txt] 

- Then run the program by typing in the command shell of the project directory

[python Simple_Naive_Bayes.py]

- It takes a minute or two to load all the files and pre process it at the same time while loading

- After the files have been loaded and processed you can see the results in csv files 
- 'prior_probability.csv' : Probabilities of the Docs for each class with respect to all docs in the sample data 
- 'word_prob_file.csv' : After training on Sample Data the Probabilities of the token or word for each Class or category is stored in this file (Trained Model file)
- 'actual_predicted_values.csv': The final result can be compared looking at this file and The accuracy percentage is printed on the terminal. 