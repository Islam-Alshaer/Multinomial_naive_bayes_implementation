import pandas as pd
import os
from MyMultiNomialNB import myMultinomialNB
import numpy as np
'''
the goal of this file is to do sentiment analysis on movie reviews using multinomial naive bayes. We will use the IMDB dataset, which is a collection of 50,000 movie reviews labeled as positive or negative. We will preprocess the data, extract features using bag-of-words, and then train a multinomial naive bayes classifier to predict the sentiment of the reviews.

1. we load the train dataset joining positive and negative
2. we preprocess the data by removing stop words, punctuation, and converting to lowercase
3. we extract features using bag-of-words, which is a representation of the text as a vector of word counts.
4. we load the test dataset
5. we predict the sentiment of the test reviews using the trained model giving it the dictionary
6. we evaluate accuracy 
'''

def load_data(root_dir):
    """
        load the train dataset and return a dataframe with two columns: 'review' and 'sentiment.'
        input is a string root directory
    """
    print('loading data from files...')
    #data are text files in the folders 'train/pos', 'train/neg'

    #empty dataframe with dtypes


    #first the positive reviews
    reviews = []
    sentiments = []

    for file in os.listdir(f'{root_dir}/pos'):
        with open(os.path.join(f'{root_dir}/pos', file), 'r') as f:
            review = f.read()
            reviews.append(review)
            sentiment = 1
            sentiments.append(sentiment)

    #then the negative reviews
    for file in os.listdir(f'{root_dir}/neg'):
        with open(os.path.join(f'{root_dir}/neg', file), 'r') as f:
            review = f.read()
            reviews.append(review)
            sentiment = 0
            sentiments.append(sentiment)

    df = pd.DataFrame({
        'review': pd.Series(reviews),
        'sentiment': pd.Series(sentiments)
    })

    print("data loaded successfully!")
    return df




def test_load_data():
    df = load_data('IMDB/train')
    print(df.head())
    print(df.shape)
    print(df['sentiment'].value_counts())


def test_count_frequencies():
    train_df = load_data('IMDB/train')
    model = myMultinomialNB()
    frequencies = model._compute_frequencies(train_df)
    print("frequency of the:", frequencies[0]['the'])
    print("frequency of are:", frequencies[1]['are'])
    print("frequency of :) emojy in positive:", frequencies[1][':)'])
    print("frequency of :) emojy in negative:", frequencies[0][':)'])
    for sentiment in frequencies:
        for token in frequencies[sentiment]:
            if frequencies[sentiment][token] <= 0:
                print('a frequency is less than or equal to 0. token',
                      token, " is: ", frequencies[sentiment][token],' something is wrong!')
                return

    print("no frequency less than 0!")


def calculate_vocab_size():
    with open('IMDB/imdb.vocab', 'r') as f:
        return len(f.read().split())

def main():
    #load train dataframe
    train_df = load_data('IMDB/train')

    #model stuff
    model = myMultinomialNB()
    vocab_size = calculate_vocab_size()
    model.fit(train_df, vocab_size, alpha=1.0)

    #load test dataset
    test_df = load_data('IMDB/test')
    y_test = test_df['sentiment']
    X_test = test_df.drop(columns=['sentiment'])

    #predict
    y_pred = model.predict(X_test)

    # print(y_pred[:100])
    # print(y_pred[-100:])

    #calculate accuracy
    print("accuracy: ", np.mean(y_pred == y_test))


def visualize():
    import matplotlib.pyplot as plt
    #visualize the most frequent 20 word for each sentiment
    train_df = load_data('IMDB/train')
    model = myMultinomialNB()
    frequencies = model._compute_frequencies(train_df)
    #get the top 20 words for each sentiment
    pos_freq = frequencies[1]
    neg_freq = frequencies[0]
    pos_top_20 = sorted(pos_freq.items(), key=lambda x: x[1], reverse=True)[:20]
    neg_top_20 = sorted(neg_freq.items(), key=lambda x: x[1], reverse=True)[:20]
    print("top 20 words in positive reviews: ", pos_top_20)
    print("top 20 words in negative reviews: ", neg_top_20)

    #plot
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.barh([x[0] for x in pos_top_20], [x[1] for x in pos_top_20], color='green')
    plt.title('Top 20 words in positive reviews')
    plt.subplot(1, 2, 2)
    plt.barh([x[0] for x in neg_top_20], [x[1] for x in neg_top_20], color='red')
    plt.title('Top 20 words in negative reviews')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # test_load_data()
    # test_count_frequencies()
    # visualize()
    main()











