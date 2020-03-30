from prepro import flatten_json, clean_spaces, normalize_text
import argparse
import logging
import nltk
import spacy
import numpy as np
import re
import six.moves.cPickle as pickle
import pycrfsuite
from sklearn.metrics import classification_report
from train import score as TScore
import random


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
        'word.ispercent=%s' % extract_percentage(word),
        'word.is_ci=%s' % is_ci(word),
        'word.isP=%s' % contain_pvalue(word),
        'word.isP2=%s' % contain_pvalue2(word),
        'word.isSigni=%s' % contain_significance(word),
        'word.isOR=%s' % contain_or(word),
        'postag=' + postag
    ]

    # Features for words that are not
    # at the beginning of a document
    if i > 0:
        word1 = doc[i - 1][0]
        postag1 = doc[i - 1][1]
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
    if i < len(doc) - 1:
        word1 = doc[i + 1][0]
        postag1 = doc[i + 1][1]
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


def contain_pvalue(word):
    if "p=" or "p>" or "p<" or "P=" or "P>" or "P<" in word:
        return True
    else:
        return False


def contain_pvalue2(word):
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


def is_ci(word):
    if "ci" or "CI" in word:
        return True
    else:
        return False


def contain_significance(word):
    if "Signi" or "signi" in word:
        return True
    else:
        return False


def contain_or(word):
    if "or" == word or "OR" == word or ("odd" or "Odd") in word \
            or ("ratio" or "Ratio") in word:
        return True
    else:
        return False


def extract_percentage(sentence):
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


def annotate(row, mode="train"):
    global nlp
    id_, context, question = row[:3]
    # c_doc = nlp(clean_spaces(context))
    # context_tokens = [normalize_text(w.text) for w in c_doc]
    context_tokens = context.split()
    # context_tokens = [w.lower() for w in context_tokens]
    answer = row[-3] if mode == "train" else row[-1][0]
    if mode != "train": print("answer: {}".format(answer))
    # print(context_tokens)
    for w in context_tokens:
        if answer in w:
            print("word: {}".format(w))
            print(answer)
    labels = ["V" if answer[0] in w else "N" for w in context_tokens]
    # print(labels)
    return list(zip(context_tokens, labels))


def construct_data(docs, mode="train"):
    docs = [annotate(doc, mode) for doc in docs]
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


def clean_non_numeric(text):
    return re.sub('[^0-9\.]', '', text)


def setup():
    parser = argparse.ArgumentParser(
        description='Preprocessing data files, about 10 minitues to run.'
    )
    parser.add_argument('--trn_file', default='HBCP/effect-all/train.effect.cand.json',
                        help='path to train file.')
    parser.add_argument('--dev_file', default='HBCP/effect-all/dev.effect.cand.json',
                        help='path to dev file.')
    parser.add_argument('--test_file', default='HBCP/effect-all/test_top1.effect.cand.json',
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


def train():
    args, log = setup()
    """initialize spacy in each process"""
    global nlp
    nlp = spacy.load('en', parser=False)

    train = flatten_json(args.trn_file, 'train')
    dev = flatten_json(args.dev_file, 'dev')
    log.info('json data flattened.')

    train_docs = construct_data(train)
    dev_docs = construct_data(dev, mode="dev")

    X_train = [extract_features(doc) for doc in train_docs]
    y_train = [get_labels(doc) for doc in train_docs]

    X_dev = [extract_features(doc) for doc in dev_docs]
    y_dev = [get_labels(doc) for doc in dev_docs]

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
        target_names=["N", "V"]))


def inference():
    args, log = setup()
    """initialize spacy in each process"""
    global nlp
    nlp = spacy.load('en', parser=False)

    test = flatten_json(args.test_file, 'test')
    log.info('json data flattened.')

    test_docs = construct_data(test, mode='test')

    X_test = [extract_features(doc) for doc in test_docs]
    y_test = [get_labels(doc) for doc in test_docs]

    tagger = pycrfsuite.Tagger()
    tagger.open('crf.model')
    predictions = []
    scores = []
    ins = 0
    # test = np.array(test)
    for xseq in X_test:
        y_pred = tagger.tag(xseq)
        y_prob = tagger.probability(y_pred)
        context = test[ins][1]
        # test_doc = nlp(clean_spaces(context))
        # test_context_tokens = [normalize_text(w.text) for w in test_doc]
        test_context_tokens = context.split()
        assert (len(test_context_tokens) == len(y_pred))
        ins += 1
        indices = [i for i, y in enumerate(y_pred) if y == 'V']
        # print(indices)
        if len(indices) == 0:
            score = 0
            pred = "null"
        else:
            score = y_prob
            # if tagged more than 1, pick randomly (only for outcome)
            # TODO: maybe choose based on confidence score
            pred = random.choice([test_context_tokens[i] for i in indices])
            # pred = ' '.join(str(test_context_tokens[i]) for i in indices)
        scores.append(score)
        predictions.append(pred)

    with open("HBCP/effect-all/test_top1.effect.dict.pkl", "rb") as f:
        cqDict = pickle.load(f)
    print(len(predictions))
    print(len(cqDict))
    assert len(predictions) == len(cqDict)

    cDict = {}
    for i in range(0, len(predictions)):
        qid = cqDict[i]
        if qid in cDict:
            cDict[qid].append(i)
        else:
            cDict[qid] = [i]
    actualPredictions = []
    actualAns = []
    actualScores = []

    for qid in cDict:
        # group by att + docid
        # and get the prediction with max score
        cList = np.array(cDict[qid])
        pScores = [scores[idx] for idx in cList]
        # bestIdx = cList[pScores.index(max(pScores))]
        # fixed! get relative indices from pScores -> get actual from cList
        bestIdx = [cList[idx] for idx in np.array(pScores).argsort()[-1:][::-1]]
        bestP = [predictions[idx] for idx in bestIdx]
        bestS = [scores[idx] for idx in bestIdx]
        ans = qid.split("\t")[-1]

        # make a HARD threshold for
        # filtering (no presence) attributes
        # only for whole pipeline evaluation
        # if (bestS[0] < 0.4): bestP = ["null"]
        actualPredictions.append(bestP)
        actualAns.append(ans)
        actualScores.append(bestS)

    for i in range(0, 10):
        print(actualPredictions[i])
        print(actualAns[i])
    # em, f1 = TScore([a.strip() for a in actualAns], [a[0].strip() for a in actualPredictions], evaluation=True)
    em, f1 = TScore(actualAns, actualPredictions, evaluation=True)
    print("dev EM: {} F1: {}".format(em, f1))


if __name__ == "__main__":
    inference()
    # train()
