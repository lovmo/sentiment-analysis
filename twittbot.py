import re
import collections
import nltk
import numpy as np
import pandas as pd
from tqdm import tqdm
from string import punctuation
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

wnl = WordNetLemmatizer()


# Importing the raw data from csv file, returning a list containing tuples. Like : [(sentiment, tweet), ...].
def clean_data(document):
    print('Cleaning the dataset...')
    raw_data = pd.read_csv(document)
    target = [sentiment.lower() for sentiment in raw_data['airline_sentiment']]  # Extracting all sentiments
    tweet_text = [clean_text(text) for text in tqdm(raw_data['text'], desc='Cleaning text')]  # Extracting all tweets

    # using ZIP to concatenate the sentiment and twitter text. Converting it to a np.array, so we can use
    # train_test_split on our dataset.
    sentiment_tweet = np.array(list(zip(target, tweet_text)))
    return sentiment_tweet


def clean_text(tweet):
    clean_tweet = re.sub('[^a-z\s]+', ' ', tweet, flags=re.IGNORECASE)
    clean_tweet = re.sub('(\s+)', ' ', clean_tweet)
    clean_tweet = nltk.word_tokenize(clean_tweet.lower())
    return [wnl.lemmatize(word) for word in clean_tweet if
            word not in stopword]  # Removing only punctuation. Not words.


# Bag of word function. Using collections module to count the frequency for each word.
def bow(text):
    word_count = collections.Counter(text)  # Can use .most_common(X) to find the most common words.
    return dict(word_count)


# Function calculate the probability for each class.
def p_classes(dataset):
    classes = [i[0] for i in dataset]
    tot_classess = len(classes)

    p_class = {
        'positive': (classes.count('positive') / tot_classess),
        'neutral': (classes.count('neutral') / tot_classess),
        'negative': (classes.count('negative') / tot_classess)
    }
    return p_class


# Function create a dictionary with all the words summed up in each class. The dictionary have a total sum for every
# word.
def data_process(data):
    # Pre defining dict items for totals.
    keywords = {
        'sum_wrd': {'positive': 0, 'neutral': 0, 'negative': 0},
        'vocabulary': []  # Is our |V|.
    }

    # Sorting all the tweets to their respecting class, adding every text together. Adding it to the keywords dict.
    for i in data:
        if i[0] not in keywords.keys():
            keywords.update({i[0]: i[1]})  # Initiating by creating the list
        else:
            keywords[i[0]].extend(i[1])  # Adding every tweet to the same list for each class.

    # Iterating over every class and using bag of word method to represent the words with numbers.
    for i in tqdm(keywords, desc='Creating vocabulary'):
        if i == 'sum_wrd' or i == 'vocabulary':  # Skipping these two.
            continue
        else:
            # Making a bag of word for each class.
            keywords[i] = bow(keywords[i])

            # Making summary-data for each class, counting how many words there is total for every class.
            for key, value in keywords[i].items():
                keywords['sum_wrd'][i] += int(value)

                # Adding every word to a list, which will become the vocabulary |v|.
                if key not in keywords['vocabulary']:
                    keywords['vocabulary'].append(key)
    return keywords


def naive_bayes(inp, explanation=False):
    score_dict = {'positive': p_class['positive'], 'neutral': p_class['neutral'], 'negative': p_class['negative']}
    scor_expl = {}

    for word in inp:
        if word in vocab:
            for sentiment in score_dict.keys():
                sum_wrd_class = (keywords['sum_wrd'][sentiment])

                # To calculate the probability for each word, we use add-1 smoothing.
                # Formula is: count(word) + 1 / (all-words-class + vocabulary)
                p_word = float(keywords[sentiment].get(word, 1) / (sum_wrd_class + len(vocab)))
                score_dict[sentiment] *= p_word

                # Allowing the explanation function. It is by default disabled.
                if explanation:
                    if word not in scor_expl.keys():
                        scor_expl.update({word: [(sentiment, p_word * p_class[sentiment])]})
                    else:
                        scor_expl[word].append((sentiment, p_word * p_class[sentiment]))

    max_class = max(score_dict, key=score_dict.get)

    if explanation:
        return max_class, scor_expl
    else:
        return max_class


def run_model():
    accuracy = {'positive': 0, 'negative': 0}
    for i in x_test:
        predicted_value = naive_bayes(i[1])
        actual_value = i[0]
        if predicted_value == actual_value:
            accuracy['positive'] += 1
        else:
            accuracy['negative'] += 1

    tot_accuracy = (accuracy['positive'] / sum(accuracy.values())) * 100
    print(f'The model is {tot_accuracy:.2f}% accurate')


# Making a user menu, enabling the user to input own tweets.
def user_exp():
    print('Welcome to the naive bayes sentiment predictor.')
    while True:
        print('\nEnter: Start\n    Q: Quit\n')
        start = input('-->').lower()
        if start != 'q':
            user_inp = clean_text(input('Add your tweet: '))
            predicted_value, explanation = naive_bayes(user_inp, explanation=True)

            print(f'\nThe tweet is {predicted_value}. The explanation for the prediction is:')
            for key, value in explanation.items():
                max_value = max(value, key=lambda y: y[1])

                print(f'{key.upper()}: Has highest probability with sentiment {max_value[0].upper()}')
                print(f' ---> P({key}|{max_value[0]}) = {max_value[1]}\n')
            print(f'The sentiment {predicted_value} is weighted most.')
            print(f'The probability for the class is {p_class[predicted_value]}.')
        else:
            break


# Initiating variables

# A list with stopwords we want to exclude from the algorithm.
stopword = set(list(punctuation) + stopwords.words('english'))

# Import the dataset and cleaning the data.
tweets = clean_data('Tweets.csv')

# Splitting the cleaned dataset into training and test sets.
x_train, x_test = train_test_split(tweets, test_size=0.20)

# Organizing and sorting every word and class. Making it possible to count and measure
keywords = data_process(x_train)

# Initiating our vocabulary.
vocab = keywords['vocabulary']

# A dictionary with the prior probability for each class.
p_class = p_classes(x_train)

run_model()

user_exp()
