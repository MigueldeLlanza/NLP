import nltk, difflib
from nltk.util import ngrams
import matplotlib.pyplot as plt
from nltk.corpus import treebank_raw, treebank_chunk
nltk.download('treebank')

pattern = r'''(?x)                   # set flag to allow verbose regexps
     (?:[A-Z]\.|[a-z]\.)+            # acronyms, e.g., U.S.A.
   | [A-Z][a-z]{,3}\.                # abbreviations, e.g., Nov.
   | \d+(?:-\w+)+                    # number-word with interval hyphen, e.g., 12-month
   | \d+[a-z]|\d+/\d+                # numbers ending in a letter, or with an internal /
   | \d+(?:\.?\,?\d+)+               # numbers with optional decimals 
   | \w+(?=n't)|n't|\w+(?=')|'\w+    # contractions
   | \w+(?:-\w+)*(?:/\w+)*           # words with optional internal hyphens
   | \.\.\.                          # ellipsis 
   | [][\.$,;"'?():-_`{}%&#]         # these are separate tokens; includes ], [
 '''

def get_corpus_text(nr_files=199):
    """Returns the raw corpus as a long string.
    'nr_files' says how much of the corpus is returned;
    default is 199, which is the whole corpus.
    """
    fileids = nltk.corpus.treebank_raw.fileids()[:nr_files]
    corpus_text = nltk.corpus.treebank_raw.raw(fileids)
    corpus_text = corpus_text.replace(".START", "")
    
    return corpus_text

def fix_treebank_tokens(tokens):
    """Replace tokens so that they are similar to the raw corpus text."""
    
    return [token.replace("''", '"').replace("``", '"').replace(r"\/", "/") for token in tokens]

def get_gold_tokens(nr_files=199):
    """Returns the gold corpus as a list of strings.
    'nr_files' says how much of the corpus is returned;
    default is 199, which is the whole corpus.
    """
    fileids = nltk.corpus.treebank_chunk.fileids()[:nr_files]
    gold_tokens = nltk.corpus.treebank_chunk.words(fileids)
    
    return fix_treebank_tokens(gold_tokens)

def tokenize_corpus(text, pattern):
    """tokenize the text input with the regular expression in pattern"""
    tokens = nltk.regexp_tokenize(text, pattern)
    
    return tokens

def evaluate_tokenization(test_tokens, gold_tokens):
    """Finds the chunks where test_tokens differs from gold_tokens.
    Prints the errors and calculates similarity measures.
    """
    matcher = difflib.SequenceMatcher()
    matcher.set_seqs(test_tokens, gold_tokens)
    error_chunks = true_positives = false_positives = false_negatives = 0
    print(" Token%30s | %-30sToken" % ("Error", "Correct"))
    print("-" * 38 + "+" + "-" * 38)
    for difftype, test_from, test_to, gold_from, gold_to in matcher.get_opcodes():
        if difftype == "equal":
            true_positives += test_to - test_from
        else:
            false_positives += test_to - test_from
            false_negatives += gold_to - gold_from
            error_chunks += 1
            test_chunk = " ".join(test_tokens[test_from:test_to])
            gold_chunk = " ".join(gold_tokens[gold_from:gold_to])
            print("%6d%30s | %-30s%d" % (test_from, test_chunk, gold_chunk,gold_from))
    precision = 1.0 * true_positives / (true_positives + false_positives)
    recall = 1.0 * true_positives / (true_positives + false_negatives)
    fscore = 2.0 * precision * recall / (precision + recall)
    print()
    print("Test size: %5d tokens" % len(test_tokens))
    print("Gold size: %5d tokens" % len(gold_tokens))
    print("Nr errors: %5d chunks" % error_chunks)
    print("Precision: %5.2f %%" % (100 * precision))
    print("Recall: %5.2f %%" % (100 * recall))
    print("F-score: %5.2f %%" % (100 * fscore))


def corpus_length(corpus):
    """
    returns the total number of tokens in the corpus
    """
    corpus_length = len(corpus)
    
    return corpus_length

def number_types(corpus):
    """
    returns the number of types in the corpus
    """
    nr_types = len(set(corpus))
    
    return nr_types

def average_length(corpus):
    """
    sum the length of all tokens and divides it by the total number of tokens in the corpus
    """
    token_length = 0
    for x in corpus:
        token_length += len(x)
    
    return token_length/len(corpus)

def longest_token(corpus):
    """
    calculate the element with maximum length, and compare this length to the length of the 
    rest of the tokens to obtain all the tokens with maximum length
    """
    token_size = max([(len(x), x) for x in corpus])
    longest_tokens = [(len(i), i) for i in corpus if len(i) == token_size[0]]
    
    return longest_tokens

def freq_dist(corpus):
    """
    calculate frequency distribution of tokens
    """
    fd = nltk.FreqDist(corpus)
    
    return fd

def hapaxes(corpus):
    """
    returns the number of hapaxes in the corpus
    """
    fd = freq_dist(corpus)
    length_hapaxes = len(fd.hapaxes()) 
    
    return length_hapaxes

def percentage(count, total):
    """
    calculates the percentage
    """
    
    return 100 * count/total


def most_frequent(corpus):
    """
    returns the 10 most frequent types
    """
    fd = nltk.FreqDist(corpus)
    
    return fd.most_common(10)

def percentage_common_types (corpus):
    """
    sum the occurrences of the 10 most frequent types and divides it by the total number of tokens in the corpus 
    """
    total = sum([t[1] for t in most_frequent(corpus)])
    
    return percentage(total, corpus_length(corpus))


def divide_corpus(corpus, number_of_partitions):
    """
    divides the corpus in 10 equally large subcorpora (0-9)
    """
    partition_length = corpus_length(corpus) / number_of_partitions
    list_of_index = []
    for i in range(number_of_partitions + 1):
        list_of_index.append(partition_length*i)
    list_of_index = [int(i) for i in list_of_index]
    ind_bigr = nltk.bigrams(list_of_index)
    corpus_parts = []
    for bigr in ind_bigr:
        corpus_parts.append(corpus[bigr[0]:bigr[1]])
    
    return corpus_parts


def hapaxes_parts(corpus_parts):
    """
    returns the number of hapaxes per partition, taking into account 
    the hapaxes already present in the previous visited subcorpora
    """
    hapaxes_found = [] 
    result = [] 
    for part in corpus_parts:
        fd = freq_dist(part)
        hapaxes_found.append(fd.hapaxes())
    for x in range(len(hapaxes_found)):
        hapaxes_found[x] = [hapax for hapax in hapaxes_found[x] if hapax not in result]
        result.extend(hapaxes_found[x])
    number_hapaxes = [len(sub_hapaxes) for sub_hapaxes in hapaxes_found]
    
    return number_hapaxes

def percentage_hapaxes(corpus_parts, corpus):
    hapax_percentage = []
    count = 0
    dv_corpus = divide_corpus(corpus, 10)
    hapax_parts = hapaxes_parts(corpus_parts)
    for x in hapax_parts:
        hapax_percentage.append(percentage(x, len(dv_corpus[count])))
        count += 1
    
    return hapax_percentage

def n_grams(corpus, ngr):
    """
    returns a tuple of the unique ngrams (depending on the input that this function receives) in the corpus and
    the percentage of unique ngrams of all the ngrams
    """
    ngram = list(ngrams(corpus, ngr))
    total_ngram = nltk.FreqDist(ngram)
    unique_ngram = [x for x in total_ngram if total_ngram[x] == 1]
    unique_ngram = len(unique_ngram)
    percentage_unique_ngram = percentage(unique_ngram, len(ngram))
    
    return (unique_ngram, percentage_unique_ngram)


def corpus_statistics(corpus, dv_corpus):
    """ 
    Call each function of the Corpus Statistics questions and print the result
    """
    
    print('There are {} tokens in the corpus, and there are {} types.\n' .format(corpus_length(corpus), number_types(corpus)))
    print('There average token length is {}.\n' .format(average_length(corpus)))
    print('The longest tokens are {}.\n' .format(longest_token(corpus)))
    print('The number of hapaxes is {} with a {}%\n.' .format(hapaxes(corpus), percentage(hapaxes(corpus), corpus_length(corpus))))
    print('The 10 most frequent types (along with their frequencies) are {}  and represent the {}% of the total tokens.\n' .format(most_frequent(corpus), percentage_common_types(corpus))) 
    print('The hapaxes for each partition (0-9) are {}.\n' .format(hapaxes_parts(dv_corpus)))
    print('The percentage of hapaxes for each partition (0-9) is {}.\n' .format(percentage_hapaxes(dv_corpus, corpus)))
    print('The unique bigrams and the percentage they represent are {}. The unique trigrams and the percentage they represent are {}.' .format(n_grams(corpus, 2), n_grams(corpus, 3)))      


if __name__ == "__main__":
    nr_files = 199
    corpus_text = get_corpus_text(nr_files)
    gold_tokens = get_gold_tokens(nr_files)
    tokens = tokenize_corpus(corpus_text, pattern)
    evaluate_tokenization(tokens, gold_tokens)
    print("\nCORPUS STATISTICS:\n")
    dv_corpus = divide_corpus(tokens, 10)
    corpus_statistics(tokens, dv_corpus)

"""
given the data obtained by the functions divide_corpus(corpus, number_of_partitions) and hapaxes_parts(corpus_parts),
the graphic for the number of hapaxes per partition is plotted
"""
hapax_parts = hapaxes_parts(dv_corpus)   
parts_length = [h for h in range(len(hapax_parts))]

fig = plt.figure()
plt.bar(parts_length, hapax_parts)
fig.suptitle('Number of hapaxes per partition', fontsize=18)
plt.xlabel('Partitions', fontsize=12)
plt.ylabel('Hapaxes', fontsize=12)
plt.show()


"""
given the data obtained by the function percentage_hapaxes(dv_corpus, tokenized_corpus),
the graphic for the percentage of hapaxes per partition is plotted
"""
per_hapaxes = percentage_hapaxes(dv_corpus, tokens)
perc_length = [p for p in range(len(per_hapaxes))]

fig = plt.figure()
plt.bar(perc_length, per_hapaxes)
fig.suptitle('Percentage of hapaxes per partition', fontsize=18)
plt.xlabel('Partitions', fontsize=12)
plt.ylabel('% hapaxes', fontsize=12)
plt.show()
