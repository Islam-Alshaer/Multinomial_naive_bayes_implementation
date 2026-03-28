import math
from collections import defaultdict
import emot
import numpy as np
import pandas as pd

emot_obj = emot.core.emot()  # we will need it later

def get_list_of_clean_subtokens(token):
    token = token.lower()

    # a token is either all alphabet or all symbols or part and part or emoji
    is_all_alpha = all(char.isalpha() for char in token)
    is_accepted_symbols = all(char in '?!' for char in token) #ex: ?????? or ??!!! or !? or ? or !
    is_emoji = bool(emot_obj.emoticons(token)['location'])  # assuming that it's all an emoji, not part and part like this: "I loved it:)" but rather : "I loved it :)"

    if is_all_alpha or is_emoji or is_accepted_symbols:
        return [token]
    else:  # we divide it into multiple words, alphabetic and symbolic
        # for example: Erika! is  going to be Erika and !
        # but paranthases and brackets will be ignored
        # also the - and ' will be added with the word as a whole new word, ex: mind-blowing
        alphabetic_part = ''
        symbolic_part = ''
        for char in token:
            if char.isalpha() or (char == '-') or (char == '\''):
                alphabetic_part += char
            elif char in '(){}[]':
                continue
            else:
                symbolic_part += char

        if symbolic_part not in '?!':
            symbolic_part = '' # we ignore symbols other than ? and !

        if alphabetic_part != '' and symbolic_part != '':
            return [alphabetic_part, symbolic_part]
        elif alphabetic_part != '':
            return [alphabetic_part]
        elif symbolic_part != '':
            return [symbolic_part]

        return None


class myMultinomialNB:

    def __init__(self):
        self.frequencies = defaultdict(lambda: defaultdict(int))
        self.alpha = None
        self.priors = defaultdict(float) #actually this is useless for this dataset
        self.vocab_size = None
        self.total_words = defaultdict(int)
        self.y_classes = None


    def _compute_frequencies(self, train_df):
        """expects a dataframe, returns a dictionary of each word from vocab's frequency in training data all in all
            please note that a word can exist in positive only or negative only
            please note that it's frequencies[sentiment][token] not the opposite
        """
        print('counting frequencies (bag of words)')

        frequencies = defaultdict(lambda: defaultdict(int))

        # iterate over each document
        for row in train_df.itertuples():
            review = row.review
            sentiment = row.sentiment
            # iterate over each word (token)
            for token in review.split():
                clean_subtokens = get_list_of_clean_subtokens(token)
                if clean_subtokens is None: continue
                #if somehow returns an empty token, we skip it
                for subtoken in clean_subtokens:
                    frequencies[sentiment][subtoken] += 1

        print('frequencies counted successfully!')
        return frequencies


    def fit(self, X_train, vocab_size, alpha=1.0):
        print("fitting...")
        self.y_classes = X_train['sentiment'].unique()
        self.total_words = X_train['sentiment'].value_counts()
        self.priors = self.total_words / len(X_train) #this is a series (can be accessed by the y_class directly)
        self.alpha = alpha
        self.vocab_size = vocab_size
        self.frequencies = self._compute_frequencies(X_train)
        print("fitting done successfully!")


    def predict(self, X):
        """
        :param X: dataframe
        :return: numpy array of predictions
        """
        print("predicting....")
        y_pred = np.array([])
        #for each example
        for example in X.itertuples():
            review = example.review
            scores = defaultdict(float)

            #for each class
            for y_class in self.y_classes:

                scores[y_class] = math.log(self.priors[y_class]) #initialize score with log prior

                #for each token
                for token in review.split():
                    subtokens = get_list_of_clean_subtokens(token)
                    if not subtokens: continue
                    for subtoken in subtokens:
                        #calculate log likelihood
                        log_likelihood = (math.log(self.frequencies[y_class][subtoken] + self.alpha)
                                      - math.log(self.total_words[y_class] + self.alpha * self.vocab_size))

                        # weight = self.frequencies[y_class][subtoken]

                        # scores[y_class] += weight * log_likelihood
                        scores[y_class] += log_likelihood
            #decide who wins
            winner_class = max(scores, key=scores.get) #the class with the maximum score
            y_pred = np.hstack([y_pred, winner_class]) #append to the prediction array

        print("prediction done successfully!")
        return y_pred