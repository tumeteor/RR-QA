from prepro import flatten_json, clean_spaces, normalize_text
import argparse
import logging
import nltk
import spacy
import numpy as np
import re
from sklearn.metrics import classification_report

def word2features(doc, i):
    word = doc[i][0]
    postag = doc[i][1]

    # Common features for all words
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'word.ispercent=%s' %extractPercentage(word),
        'word.isCI=%s' %isCI(word),
        'word.isP=%s' %containPvalue(word),
        'word.isP2=%s' % containPvalue2(word),
        'word.isSigni=%s' %containSignificance(word),
        'word.isOR=%s' %containOR(word),
        'postag=' + postag
    ]

    # Features for words that are not
    # at the beginning of a document
    if i > 0:
        word1 = doc[i-1][0]
        postag1 = doc[i-1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:word.isdigit=%s' % word1.isdigit(),
            '-1:postag=' + postag1
        ])
    else:
        # Indicate that it is the 'beginning of a document'
        features.append('BOS')

    # Features for words that are not
    # at the end of a document
    if i < len(doc)-1:
        word1 = doc[i+1][0]
        postag1 = doc[i+1][1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:word.isdigit=%s' % word1.isdigit(),
            '+1:postag=' + postag1
        ])
    else:
        # Indicate that it is the 'end of a document'
        features.append('EOS')

    return features
def containPvalue(word):
    if "p=" or "p>" or "p<" or "P=" or "P>" or "P<" in word:
        return True
    else:
        return False

def containPvalue2(word):
    pvalues = []
    # look for -, a digit, a dot ending with a digit and a percentage sign
    rx = r'[pP]\s*[=<>]\s*[-+]?\d*\.\d+|\d+'
    # loop over the results
    for match in re.finditer(rx, word):
        interval = match.group(0).split('-')
        for pvalue in interval:
            if pvalue.startswith("p") or pvalue.startswith("P"):
                pvalues.append(pvalue)
    if len(pvalues) > 0:
        return True
    else:
        return False


def isCI(word):
    if "ci" or "CI" in word:
        return True
    else:
        return False

def containSignificance(word):
    if "Signi" or "signi" in word:
        return True
    else:
        return False

def containOR(word):
    if "or" == word or "OR" == word or ("odd" or "Odd") in word \
            or ("ratio" or "Ratio") in word:
        return True
    else:
        return False

def extractPercentage(sentence):
    numbers = []
    # look for -, a digit, a dot ending with a digit and a percentage sign
    rx = r'[-\d.]+\d%'
    # loop over the results
    for match in re.finditer(rx, sentence):
        interval = match.group(0).split('-')
        for number in interval:
            try:
                if 0 <= float(number.strip('%')):
                    numbers.append(number)
            except ValueError:
                pass

    if len(numbers) > 0:
        return True
    else:
        return False

def annotate(row):
    global nlp
    id_, context, question = row[:3]
    c_doc = nlp(clean_spaces(context))
    context_tokens = [normalize_text(w.text) for w in c_doc]
    #context_tokens = [w.lower() for w in context_tokens]
    answer  = row[-3].lower()
    labels = ["V" if answer in w or w in answer else "N" for w in context_tokens]
    return list(zip(context_tokens, labels))

def constructData(docs):
    docs = [annotate(doc) for doc in docs]
    data = []
    for i, doc in enumerate(docs):
        # Obtain the list of tokens in the document
        tokens = [t for t, label in doc]

        # Perform POS tagging
        tagged = nltk.pos_tag(tokens)

        # Take the word, POS tag, and its label
        data.append([(w, pos, label) for (w, label), (word, pos) in zip(doc, tagged)])
    return data


nlp = None



def setup():
    parser = argparse.ArgumentParser(
        description='Preprocessing data files, about 10 minitues to run.'
    )
    parser.add_argument('--trn_file', default='HBCP/effect-all/train.effect.cand.json',
                        help='path to train file.')
    parser.add_argument('--dev_file', default='HBCP/effect-all/dev.effect.cand.json',
                        help='path to dev file.')
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG,
                        datefmt='%m/%d/%Y %I:%M:%S')
    log = logging.getLogger(__name__)
    log.info(vars(args))
    log.info('start data preparing...')

    return args, log

# A function for extracting features in documents
def extract_features(doc):
    return [word2features(doc, i) for i in range(len(doc))]

# A function fo generating the list of labels for each document
def get_labels(doc):
    return [label for (token, postag, label) in doc]

def main():
    args, log = setup()
    """initialize spacy in each process"""
    global nlp
    nlp = spacy.load('en', parser=False)

    train = flatten_json(args.trn_file, 'train')
    dev = flatten_json(args.dev_file, 'dev')
    log.info('json data flattened.')

    train_docs = constructData(train)
    dev_docs = constructData(dev)

    X_train = [extract_features(doc) for doc in train_docs]
    y_train = [get_labels(doc) for doc in train_docs]

    X_dev = [extract_features(doc) for doc in dev_docs]
    y_dev = [get_labels(doc) for doc in dev_docs]

    import pycrfsuite
    trainer = pycrfsuite.Trainer(verbose=True)

    # Submit training data to the trainer
    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)

    # Set the parameters of the model
    trainer.set_params({
        # coefficient for L1 penalty
        'c1': 0.1,

        # coefficient for L2 penalty
        'c2': 0.01,

        # maximum number of iterations
        'max_iterations': 200,

        # whether to include transitions that
        # are possible, but not observed
        'feature.possible_transitions': True
    })

    # Provide a file name as a parameter to the train function, such that
    # the model will be saved to the file when training is finished
    trainer.train('crf.model')

    tagger = pycrfsuite.Tagger()
    tagger.open('crf.model')
    y_pred = [tagger.tag(xseq) for xseq in X_dev]

    # Create a mapping of labels to indices
    labels = {"V": 1, "N": 0}

    # Convert the sequences of tags into a 1-dimensional array
    predictions = np.array([labels[tag] for row in y_pred for tag in row])
    truths = np.array([labels[tag] for row in y_dev for tag in row])

    # Print out the classification report
    print(classification_report(
        truths, predictions,
        target_names=["V", "N"]))



if __name__== "__main__":
    main()