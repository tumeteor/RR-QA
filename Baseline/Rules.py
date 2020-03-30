# coding: utf-8
import re
import random


def extract_pvalue(sentence):
    pvalues = []
    rx = r'[pP]\s*[=<>]\s*[-+]?\d*\.\d+|\d+'
    # loop over the results
    for match in re.finditer(rx, sentence):
        interval = match.group(0).split('-')
        for pvalue in interval:
            if pvalue.startswith("p") or pvalue.startswith("P"):
                pvalues.append(pvalue)
    return pvalues


def extract_ci(sentence):
    cis = []
    rx = r'(?<=95% ci|95% CI)\s*.[(|[]?[\d+\.\d+]+\s?[-|–]?\s?[\d+\.\d+]+[]|)]?'
    # loop over the results
    for m in re.finditer(rx, sentence):
        match = m.group(0)
        print(match)
        match = match.replace("[", "")
        match = match.replace("]", "")
        match = match.replace("(", "")
        match = match.replace(")", "")
        match = match.strip()
        cis.append(match)
    return cis


def extract_upper_ci(sentence):
    uppers = []
    lowers = []
    cis = extract_ci(sentence)

    for ci in cis:
        try:
            if "-" in ci:
                uppers.append(float(ci.split("-")[1]))
                lowers.append(float(ci.split("-")[0]))
            elif "–" in ci:
                uppers.append(float(ci.split("–")[1]))
                lowers.append(float(ci.split("–")[0]))
            else:
                uppers.append(float(ci))
                lowers.append(float(ci))
        except ValueError:
            continue

    return lowers, uppers


def extract_or(sentence):
    cis = []
    rx = r'(OR|Odd ratio| odd ratio | Odd Ratio | Odds ratio)'
    # loop over the results
    for match in re.finditer(rx, sentence):
        interval = match.group(0).split('-')
        for ci in interval:
            cis.append(ci)
    return cis


random.seed(1013)
with open("effect.out", "r") as f:
    em = 0
    total = 0
    uniq = 0
    for line in f:
        parts = line.split("\t")
        if not parts[0].startswith("content"):
            gt = parts[0]
            qid = parts[1]
            if len(candidate_snt) == 0:
                total += 1
                continue
            else:
                # extract candidate answers
                if qid == "3870686":
                    candidate_ans = extract_or(candidate_snt)
                elif qid == "3870696":
                    candidate_ans = extract_pvalue(candidate_snt)
                elif qid == "3870697":
                    candidate_ans = extract_upper_ci(candidate_snt)[0]
                elif qid == "3870702":
                    candidate_ans = extract_upper_ci(candidate_snt)[1]
                else:
                    total += 1
                    continue
                if len(candidate_ans) == 0:
                    total += 1
                    continue

                p = candidate_ans[0]
                # p = random.choice(candidates)
                if (p == gt): em += 1
                total += 1
        else:
            candidate_snt = parts[1]
            # if len(candidates) == 0:
            #     print(line)
            #     total += 1
            #     continue

    print("total: {}".format(total))
    prec = 100. * em / total
    recall = 100. * em / 98
    print("number of unique candidates: {}".format(uniq))
    print("precision: {}".format(prec))
    print("recall: {}".format(recall))
    print("f1: {}".format((2 * prec * recall) / (prec + recall)))
