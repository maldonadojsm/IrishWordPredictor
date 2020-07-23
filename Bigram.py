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
total_words = len(clean_txt)

print(total_words)

# BIGRAM COUNTING

# Generate all potential bigram found in corpus
bi_gram = nltk.bigrams(clean_txt.split())
# Count frequency of bigram in corpus
count_freq_bigram = nltk.ConditionalFreqDist(bi_gram)
# Generate Dictionary Keys
count_freq_bigram.conditions()



# COUNT UNIQUE WORDS

unique_words = len(set(clean_txt))


test_file = open("test.txt", "r+", encoding="utf8")
test_lines = test_file.readlines()


"""
EXAMPLE:

seo {a|á} dhéanann -> seo(N-1) {a|á}(N) dhéanann(N+1)
"""


index = 0
with open("result.csv", mode='w') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Id', 'Expected'])
    j = 1

    for a, b, c, d in zip(range(len(ascii)), range(len(tilde)), range(len(wordPairs)),range(len(wordPairs2))):
        for i in test_lines:
            if wordPairs[c] in i:
                sentence = i.lower().translate(str.maketrans('', '', string.punctuation)).split()

                # TRY CAPTURING BEFORE AND AFTER WORD
                try:
                    index_plus_1 = sentence[sentence.index(wordPairs2[d]) + 1].translate(str.maketrans('', '', string.punctuation))
                    index_min_1 = sentence[sentence.index(wordPairs2[d]) - 1].translate(
                        str.maketrans('', '', string.punctuation))

                    # UNIGRAM

                    unigram_ascii = (freq_unigram[ascii[a]] + 1) / (total_words + unique_words)

                    unigram_tilde = (freq_unigram[tilde[b]] + 1) / (total_words + unique_words)

                    # NORMALIZE
                    unigram_prob = unigram_ascii / (unigram_ascii + unigram_tilde)

                    # BIGRAM

                    # IF DIV/0 Doesn't Exist
                    try:
                        # Unigram Prob of N-1

                        # SENTENCE 1: ASCII
                        # Unigram Prob of N

                        unigram_prob_ascii = freq_unigram[index_min_1] / (total_words + unique_words)

                        # Bigram Prob of N

                        left_context_ascii = (count_freq_bigram[index_min_1][ascii[a]] + 1) / (freq_unigram[index_min_1] + unique_words)

                        right_context_ascii = (count_freq_bigram[ascii[a]][index_plus_1] + 1) / (freq_unigram[ascii[a]] + unique_words)

                        sentence_1 = unigram_prob_ascii * left_context_ascii * right_context_ascii


                        # SENTENCE 2: TILDE

                        # Unigram Prob of M
                        unigram_prob_tilde = freq_unigram[index_min_1] / (total_words + unique_words)

                        # Bigram

                        left_context_tilde = (count_freq_bigram[index_min_1][tilde[b]] + 1) / (freq_unigram[index_min_1] + unique_words)

                        right_context_tilde = (count_freq_bigram[tilde[b]][index_plus_1] + 1) / (freq_unigram[tilde[b]] + unique_words)

                        sentence_2 = unigram_prob_tilde * left_context_tilde * right_context_tilde

                        # NORMALIZE

                        bigram_prob = sentence_1 / (sentence_1 + sentence_2)

                        if bigram_prob == 1 or bigram_prob == 0:
                            csv_writer.writerow([j, unigram_prob])

                        else:
                            csv_writer.writerow([j, bigram_prob])

                        #WRITE PREDICTION

                        csv_writer.writerow([j, (sentence_1 + unigram_prob) / 2])

                    # IF DENOM 0:
                    except:
                        csv_writer.writerow([j, unigram_prob])

                # IF N+1 Doesn't Exist, JUST CAPTURE BEFORE
                except:

                    bi_gram_prob = 0
                    index_min_1 = sentence[sentence.index(wordPairs2[d]) - 1].translate(str.maketrans('', '', string.punctuation))
                    # UNIGRAM
                    unigram_prob = freq_unigram[ascii[a]] / (freq_unigram[ascii[a]] + freq_unigram[tilde[b]])
                    # BIGRAM



                    try:


                        # SENTENCE 1:
                        # Unigram Prob of N
                        unigram_prob_ascii = freq_unigram[ascii[a]] / (freq_unigram[ascii[a]] + freq_unigram[tilde[b]])

                        left_context_ascii = count_freq_bigram[index_min_1][ascii[a]] / (count_freq_bigram[index_min_1][ascii[a]] + count_freq_bigram[index_min_1][tilde[b]])

                        sentence_1 = unigram_prob_ascii * left_context_ascii

                        # SENTNCE 2:

                        unigram_prob_tilde = freq_unigram[tilde[b]] / (freq_unigram[ascii[a]] + freq_unigram[tilde[b]])

                        left_context_tilde = count_freq_bigram[index_min_1][tilde[b]] / (count_freq_bigram[index_min_1][tilde[b]] + count_freq_bigram[index_min_1][ascii[a]])

                        sentence_2 = unigram_prob_tilde * left_context_tilde



                        csv_writer.writerow([j, (sentence_1 + unigram_prob) / 2])

                    except:

                        bi_gram = unigram_prob
                        csv_writer.writerow([j, bi_gram])
                j += 1


csv_file.close()
train_file.close()
test_file.close()









