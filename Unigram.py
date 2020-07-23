# !/usr/bin/env python
# title           :Unigram.py
# description     :Unigram Model without
# author          :Juan Maldonado
# date            :5/3/19
# version         :1.0
# usage           :python3 Unigram.py
# notes           :
# python_version  :3.6.5
# =================================================================================================================

import string
import nltk
import csv

ascii = ['a', 'ais', 'aisti', 'ait', 'ar', 'arsa', 'ban', 'cead', 'chas', 'chuig', 'dar', 'do',
                  'gaire', 'i', 'inar', 'leacht', 'leas', 'mo', 'na', 'os', 're', 'scor', 'te', 'teann', 'thoir']

tilde = ['á', 'áis', 'aistí', 'áit', 'ár','ársa', 'bán', 'céad','chás','chúig', 'dár', 'dó','gáire', 'í', 'inár', 'léacht', 'léas', 'mó', 'ná', 'ós', 'ré', 'scór', 'té', 'téann', 'thóir']


wordPairs = ['{a|á}', '{ais|áis}', '{aisti|aistí}', '{ait|áit}', '{ar|ár}','{arsa|ársa}', '{ban|bán}', '{cead|céad}','{chas|chás}',
                 '{chuig|chúig}', '{dar|dár}', '{do|dó}','{gaire|gáire}', '{i|í}', '{inar|inár}', '{leacht|léacht}', '{leas|léas}',
                 '{mo|mó}', '{na|ná}', '{os|ós}', '{re|ré}', '{scor|scór}', '{te|té}', '{teann|téann}', '{thoir|thóir}']

wordPairs2 = ['aá', 'aisáis', 'aistiaistí', 'aitáit', 'arár', 'arsaársa', 'banbán', 'ceadcéad', 'chaschás',
              'chuigchúig', 'dardár', 'dodó', 'gairegáire', 'ií', 'inarinár', 'leachtléacht', 'leasléas','momó', 'naná', 'osós',
              'reré', 'scorscór', 'teté', 'teanntéann', 'thoirthóir']

##### NLTK #########################

# PRE PROCESSING TRAIN TXT
train_file = open("train.txt", "r+", encoding="utf8")
train_txt = train_file.read().lower()
clean_txt = train_txt.translate(str.maketrans('', '', string.punctuation))

# UNIGRAM COUNTING
freq_unigram = nltk.FreqDist(clean_txt.split())
# Build probability distrubtion for unigrams
cprob_u_l = nltk.MLEProbDist(freq_unigram)


# BIGRAM COUNTING

# Generate all potential bigram found in corpus
bi_gram = nltk.bigrams(clean_txt.split())
# Count frequency of bigrams in corpus
count_freq_bigram = nltk.ConditionalFreqDist(bi_gram)
count_freq_bigramT = nltk.FreqDist(bi_gram)
# Generate Dictionary Keys
count_freq_bigram.conditions()



# Generate Conditional Probabilities for Bigrams
# Laplace Prob Distribution
cprob_laplace = nltk.ConditionalProbDist(count_freq_bigram, nltk.MLEProbDist)

# COUNT UNIQUE WORDS

unique_words = len(set(clean_txt))


# TEST FILE
test_file = open("test.txt", "r+", encoding="utf8")
test_lines = test_file.readlines()

with open("result.csv", mode='w') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Id', 'Expected'])
    j = 1
    for a, b, c, d in zip(range(len(ascii)), range(len(tilde)), range(len(wordPairs)), range(len(wordPairs2))):
        for i in test_lines:
            if wordPairs[c] in i:

                punctuations = '''—|!’‘()-[]{};“:'"•–=\,<>./?@#$%^”&*_~'''


                # To take input from the user
                # my_str = input("Enter a string: ")

                # remove punctuation from the string
                no_punct = ""
                for char in i:
                    if char not in punctuations:
                        no_punct = no_punct + char
                sentence = no_punct.lower().split()
                hh = sentence[sentence.index(wordPairs2[d]) - 1]
                index_min_1 = hh.translate(str.maketrans('', '', string.punctuation))


                # UNIGRAM

                # Laplace
                unigram_ascii_l = cprob_u_l.prob(ascii[a])

                unigram_tilde_l = cprob_u_l.prob(tilde[b])

                # NORMALIZE
                laplace_prob = unigram_ascii_l / (unigram_ascii_l + unigram_tilde_l)

                unigram_prob = (laplace_prob)


                csv_writer.writerow([j, unigram_prob])

                j += 1


csv_file.close()
train_file.close()
test_file.close()
