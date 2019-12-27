import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    sequences = test_set.get_all_sequences()
    Xlengths = test_set.get_all_Xlengths()
    for sequence in sequences:
        guess = None
        best_score = float('-inf')
        prob = {}
        X, lengths = Xlengths[sequence]
        for word, model in models.items():
            try:
                score = model.score(X, lengths)
            except :
                score = float('-inf')
            prob[word] = score
            if score > best_score:
                best_score = score
                guess = word
        probabilities.append(prob)
        guesses.append(guess)

    return probabilities, guesses
    
