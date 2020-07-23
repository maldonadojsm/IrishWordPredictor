# !/usr/bin/env python
# title           :BigramRCvsLCKatzLidstoneSmoothing.py
# description     :Bigram Model N-1 vs N+1 + Katz Smoothing + Lidstone Smoothing (Gamma = 0.2)
# author          :Juan Maldonado
# date            :5/2/19
# version         :1.0
# usage           :python3 BigramRCvsLCKatzLidstoneSmoothing.py
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

####### UNIGRAMS ########

# Build frequency distribution for all unigrams in corpus
freq_unigram = nltk.FreqDist(clean_txt.split())
# Build probability distribution for unigrams
cprob_u_l = nltk.LidstoneProbDist(freq_unigram,0.2)


### BIGRAMS ###############

# Generate All Possible Bigrams given Corpus
bi_gram = nltk.bigrams(clean_txt.split())
# Build conditional frequency distribution for said bigrams
count_freq_bigram = nltk.ConditionalFreqDist(bi_gram)
# Generate Dictionary Keys
count_freq_bigram.conditions()
# Generate Probability Distribution (Lidstone Smoothing)
cprob_laplace = nltk.ConditionalProbDist(count_freq_bigram, nltk.LidstoneProbDist,0.2)


# TEST FILE
test_file = open("test.txt", "r+", encoding="utf8")
test_lines = test_file.readlines()

"""
EXAMPLE:

seo {a|á} dhéanann -> seo(N-1) {a|á}(N) dhéanann(N+1)
"""

with open("result.csv", mode='w') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Id', 'Expected'])
    j = 1
    for a, b, c, d in zip(range(len(ascii)), range(len(tilde)), range(len(wordPairs)), range(len(wordPairs2))):
        for i in test_lines:
            if wordPairs[c] in i:

                # CLEAN LINE OF PUNCTUATION, MAKE LOWERCASE
                punctuations = '''—|!’‘()-[]{};“:'"•–=\,<>./?@#$%^”&*_~'''
                no_punct = ""
                for char in i:
                    if char not in punctuations:
                        no_punct = no_punct + char
                sentence = no_punct.lower().split()

                # Capture N - 1
                hh = sentence[sentence.index(wordPairs2[d]) - 1]
                index_min_1 = hh.translate(str.maketrans('', '', string.punctuation))


                # UNIGRAM PROBABILITIES

                unigram_ascii_l = cprob_u_l.prob(ascii[a])

                unigram_tilde_l = cprob_u_l.prob(tilde[b])

                # NORMALIZE
                laplace_prob = unigram_ascii_l / (unigram_ascii_l + unigram_tilde_l)

                unigram_prob = laplace_prob

                # BIGRAM
                try:
                    # UNIGRAM PROB OF N - 1
                    unigram_index_min_1 = cprob_u_l.prob(index_min_1)

                    # P(N-1, N)

                    prob_ascii_lc = unigram_index_min_1 * cprob_laplace[index_min_1].prob(ascii[a])

                    prob_tilde_lc = unigram_index_min_1 * cprob_laplace[index_min_1].prob(tilde[b])


                    # NORMALIZE
                    bigram_prob_lc = prob_ascii_lc / (prob_ascii_lc + prob_tilde_lc)


                    # TRY N-1 vs N+1
                    try:
                        # Capture N + 1
                        aa = sentence[sentence.index(wordPairs2[d]) + 1]

                        # N + 1
                        index_plus_1 = aa.translate(str.maketrans('', '', string.punctuation))

                        # P(N,N+1)

                        prob_ascii_rc = unigram_ascii_l * cprob_laplace[ascii[a]].prob(index_plus_1)

                        prob_tilde_rc = unigram_tilde_l * cprob_laplace[tilde[b]].prob(index_plus_1)

                        # NORMALIZE

                        bigram_prob_rc = prob_ascii_rc / (prob_ascii_rc + prob_tilde_rc)


                        # N-1 vs N+1 Continued
                        if bigram_prob_rc == 1 or bigram_prob_rc == 0:

                            if bigram_prob_lc == 1 or bigram_prob_lc == 0:
                                # BACK OFF TO UNIGRAM
                                csv_writer.writerow([j, unigram_prob])
                                j += 1
                            # BIGRAM N-1
                            else:
                                csv_writer.writerow([j, bigram_prob_lc])
                                j += 1
                        # WE CAN PERFORM COMPARISON
                        else:
                            # USE BIGRAM N+1 if N-1 PROB IS 0
                            if bigram_prob_lc == 1 or bigram_prob_lc == 0:
                                csv_writer.writerow([j, bigram_prob_rc])
                                j += 1
                            # PERFORM COMPARISON
                            else:
                                # USE N-1 PROB if greater than N+1
                                if bigram_prob_lc > bigram_prob_rc:
                                    csv_writer.writerow([j, bigram_prob_lc])
                                    j += 1
                                # USE N+1 PROB if greater than N-1
                                if bigram_prob_lc < bigram_prob_rc:
                                    csv_writer.writerow([j, bigram_prob_rc])
                                    j += 1

                    except:
                        # BACK OFF
                        if bigram_prob_lc == 1 or bigram_prob_lc == 0:
                            csv_writer.writerow([j, unigram_prob])
                            j += 1
                        # BIGRAM
                        else:
                            csv_writer.writerow([j, bigram_prob_lc])
                            j += 1
                # UNIGRAM
                except:
                    csv_writer.writerow([j, unigram_prob])
                    j += 1



csv_file.close()
train_file.close()
test_file.close()
