import nltk
from nltk.corpus import brown
from prettytable import PrettyTable
nltk.download('brown')
nltk.download('universal_tagset')

def print_brown_statistics(list_of_genres):
    """
    returns all the statistics for the 'universal' and 'brown' tagsets
    """
    sents_U = []
    s_U = []
    words_U = []
    w_U = []
    words_O = []
    sent_length = []
    words_length = []
    total_tags_U = []
    total_tags_O = []
    for genre in list_of_genres:
        tag_sent_U = nltk.corpus.brown.tagged_sents(categories=genre, tagset='universal')
        tag_word_U = nltk.corpus.brown.tagged_words(categories=genre, tagset='universal')
        tag_word_O = nltk.corpus.brown.tagged_words(categories=genre)
        sents_U.append(tag_sent_U)
        words_U.append(tag_word_U)
        words_O.append(tag_word_O)
        s_U.append(len(tag_sent_U))
        w_U.append(len(tag_word_U))
    
    for i in sents_U:
        sent_len_sum = 0
        for x in i:
            sent_len_sum += len(x)
        sent_length.append(round(sent_len_sum / len(i), 2))
    
    for j in words_U:
        words_len_sum = 0
        tags_U = []
        for (word, tag) in j:
            words_len_sum += len(word)
            tags_U.append(tag)
        tags_U = list(set(tags_U))
        total_tags_U.append(len(tags_U))
        words_length.append(round(words_len_sum / len(j), 2))
    
    for t in words_O:
        tags_O = []
        for (word, tag) in t:
            tags_O.append(tag)
        total_tags_O.append(len(set(tags_O)))
    
    return [total_tags_U, total_tags_O, s_U, w_U, sent_length, words_length]


def part1():
    """
    prints a table with the results of part1
    """
    brown_stats = PrettyTable()
    brown_stats.field_names = ["Brown genre", "Uni POS tags", "Orig POS tags", "Sentences", "Words", "Sent. length", "Word length"]
    genres = ["fiction", "government", "news", "reviews"]
    stats = print_brown_statistics(genres)
    
    for i in range(len(genres)):
        brown_stats.add_row([genres[i], stats[0][i], stats[1][i], stats[2][i], stats[3][i], stats[4][i], stats[5][i]])  
 
    print(brown_stats)


# PART 2
def common_ngrams(genre, d, n_results):
    """
    returns the frequency and accumulated frequency of the n most common ngrams in the 'news' genre corpus
    """
    
    for n in range(1, d + 1):
        corpus = brown.tagged_sents(categories=genre, tagset='universal')
        fd = nltk.FreqDist()
        results = []
        for s in corpus:
            tags = [tag for (word, tag) in s]
            ngrams = nltk.ngrams(tags, n, pad_left=True, pad_right=True, left_pad_symbol="$", right_pad_symbol="$")
            for ngram in ngrams:
                fd[ngram] += 1
        total_ngrams = fd.N()
        m_common = fd.most_common(n_results)
        accum_freq = 0
        for item in m_common:
            freq = round(item[1] * 100 / total_ngrams, 2)
            accum_freq += freq
            results.append((item[0], freq, accum_freq))
    
    return results


def part2():
    """
    prints a table with the results of part2
    """
    unigrams = common_ngrams('news', 1, 10)
    unigram = PrettyTable()
    unigram.field_names = ["1-gram", "Frequency", "Accum.freq."]
    
    for i in range(len(unigrams)):
        unigram.add_row([unigrams[i][0], unigrams[i][1], unigrams[i][2]])
    
    print(unigram)
    
    bigrams = common_ngrams('news', 2, 10)
    bigram = PrettyTable()
    bigram.field_names = ["2-gram", "Frequency", "Accum.freq."]
    
    for i in range(len(bigrams)):
        bigram.add_row([bigrams[i][0], bigrams[i][1], bigrams[i][2]])    
    
    print(bigram)

    trigrams = common_ngrams('news', 3, 10)  
    trigram = PrettyTable()
    trigram.field_names = ["3-gram", "Frequency", "Accum.freq."]
    
    for i in range(len(trigrams)):
        trigram.add_row([trigrams[i][0], trigrams[i][1], trigrams[i][2]])
        
    print(trigram)



# PART 3
def split_corpus(genre):
    """
    given a genre, returns the corpus splitted in the tuple (train, test)
    """
    news_sents = brown.tagged_sents(categories=genre, tagset='universal') 
    test = news_sents[:500]
    training = news_sents[500:]
    
    return (training, test)


def most_frequent_tag(data):
    """
    given a set of tagged sentences it extracts the most frequent tag
    input expected: list of sentences (list of tuples)
    """
    tags = []
    for x in data:
        for (word, tag) in x:
            tags.append(tag)
    
    return nltk.FreqDist(tags).max()


def train_nltk_taggers(train, test):
    """
    trains and avaluate a bigram tagger (with backoff cascade)
    input expected: train and test data (list of sentences (list of tuples))
    """
    defaultTagger = nltk.DefaultTagger(most_frequent_tag(train))
    affixTagger = nltk.AffixTagger(train, backoff=defaultTagger)
    uniTagger = nltk.UnigramTagger(train, backoff=affixTagger)
    biTagger = nltk.BigramTagger(train, backoff=uniTagger)
    triTagger = nltk.TrigramTagger(train, backoff=biTagger)
    
    return (defaultTagger, affixTagger, uniTagger, biTagger, triTagger)


def evaluate_taggers(train, test):
    """
    returns the accuracy and errors for a given train and test corpus
    """
    taggers = train_nltk_taggers(train, test)
    eva_tag = []
    for tag in taggers:
        eva_tag.append(round(tag.evaluate(test) * 100, 2))
    
    errors = []
    for a in eva_tag:
        errors.append(round(100 / (100 - a), 1))

    return (eva_tag, errors)


def part3():
    """
    prints a table with the results of part3
    """
    
    train, test = split_corpus('news')
    accu_taggers = evaluate_taggers(train, test)
    accu_tagger = PrettyTable()
    taggers = ['default', 'affix', 'unigram', 'bigram', 'trigram']
    accu_tagger.field_names = ["Genre: news", "Accuracy", "Errors (words/error)"]
    
    for i in range(len(taggers)):
        accu_tagger.add_row([taggers[i], accu_taggers[0][i], accu_taggers[1][i]])    
    
    print(accu_tagger)


# PART 4
def biTag(train, test):
    """
    returns the accuracy and errors of the biTagger for any (tran, test) tuple 
    """
    return (evaluate_taggers(train, test)[0][3], evaluate_taggers(train, test)[1][3])


def test_training(train, test):
    """
    returns accuracy and errors of the biTagger for the tuples (tran, test) and (train, train) of the 'news' genre
    """
    train_test = biTag(train, test)
    train_train = biTag(train, train)
    
    return (train_test, train_train)


def test_different_genres(list_of_genres):
    """
    returns accuracy and errors of the biTagger trained with the 'news' genre and tested with each genre
    """
    accuracy = []
    train = split_corpus('news')[0]
    for genre in list_of_genres:
        test = split_corpus(genre)[1]
        accuracy.append(biTag(train, test))
    
    return accuracy


def train_different_sizes(sizes):
    """
    returns accuracy and errors of the biTagger for different corpus size trained and tested with the 'news' genre
    """
    accuracy = []
    for size in sizes:
        train, test = split_corpus('news')
        size = int(size * len(train) / 100) 
        train_size = train[0:size]
        accuracy.append(biTag(train_size, test))
    
    return accuracy


def compare_train_test_partitions(genre):
    """
    returns accuracy and errors of the biTagger trained and tested with the 'news' genre for different partitions
    """
    train, test = split_corpus(genre)
    new_sents = brown.tagged_sents(categories=genre, tagset='universal')
    new_train = new_sents[:-500]
    new_test = new_sents[-500:]
    accuracy = biTag(train, test)
    new_accuracy = biTag(new_train, new_test)
    
    return [accuracy, new_accuracy]


def splitting_training_testing(data):
    """
    splits the input data in the tuple (train, test)
    """
    train = data[500:]
    test = data[:500]
    
    return (train, test)


def dic_simple_tags():
    """
    returns a new set of syntactic rules
    """
    dic = nltk.Index([('NOUN', 'N'), ('NUM', 'N'), ('ADJ', 'NP'), ('DET', 'NP') ,('PRON', 'NP'), ('VERB', 'V'), ('ADP', 'AUX'), ('ADV', 'AUX'), ('CONJ', 'AUX'), ('PRT', 'AUX'), ('X', 'AUX'), ('.', 'DELIM')])
    
    return dic


def super_simple_tag(sentsU):
    """
    given a data set tagged with the universal tagset, 
    returns a data set with a supersimple tagset
    """
    dic = dic_simple_tags()
    newCorpus = []
    for s in sentsU:
        newSent = []
        for t in s:
            newSent.append((t[0], dic[t[1]][0]))
        newCorpus.append(newSent)
    
    return newCorpus


def setting_data(genre):
    """
    given a brown genre, extracts and prepares the training and testing data (to build taggers)
    outputs three data sets (three tuples), containing each:
        - train set
        - testing set
    """
    newsB = brown.tagged_sents(categories=genre, tagset='brown')
    trainB, testB = splitting_training_testing(newsB)
    newsU = brown.tagged_sents(categories=genre, tagset='universal')
    trainU, testU = splitting_training_testing(newsU)
    newsS = super_simple_tag(newsU)
    trainS, testS = splitting_training_testing(newsS)
    
    return (trainB, trainU, trainS, testB, testU, testS)


def compare_different_tagsets(genre):
    """
    returns accuracy and errors for different tagsets (brown, universal and a simple tagset)
    """
    trainB, testB = setting_data(genre)[0], setting_data(genre)[3]
    trainU, testU = setting_data(genre)[1], setting_data(genre)[4]
    trainS, testS = setting_data(genre)[2], setting_data(genre)[5]
    accuracy_U = biTag(trainU, testU)
    accuracy_S = biTag(trainS, testS)
    accuracy_B = biTag(trainB, testB)
    
    return [accuracy_U, accuracy_S, accuracy_B]


def n_tags(genre):
    """
    return the number of tags (types) of each tagset
    """
    U_sents = brown.tagged_sents(categories=genre, tagset='universal')
    S_sents = super_simple_tag(U_sents)
    B_sents = brown.tagged_sents(categories=genre, tagset='brown')
    sents = [U_sents, S_sents, B_sents]
    total_tags = []
    for x in range(len(sents)):
        tags = []
        for s in sents[x]:
            for (word, tag) in s: 
                tags.append(tag)
        total_tags.append(len(set(tags)))
    
    return total_tags


def part4():
    """
    prints tables with the results of each part of part4
    """
    # Part 4a
    train, test = split_corpus('news')
    test_training(train, test)
    part_4a = test_training(train, test)
    p4a = PrettyTable()
    p4a.field_names = ["Train sentences", "Accuracy", "Errors (words/error)", "Test sentences"]
    column = ['news-train', 'news-test', 'news-train', 'news-train']
    
    for i in range(len(part_4a)):
        p4a.add_row([column[i], part_4a[i][0], part_4a[i][1], column[i]])
        
    print(p4a)
    
    # Part 4b
    genres = ["fiction", "government", "news", "reviews"]
    part_4b = test_different_genres(genres)
    p4b = PrettyTable()
    p4b.field_names = ["Train sentences", "Accuracy", "Errors (words/error)", "Test sentences"]
    column = ["fiction-test", "government-test", "news-test", "reviews-test"]
    
    for i in range(len(part_4b)):
        p4b.add_row(["news-train", part_4b[i][0], part_4b[i][1], column[i]])
        
    print(p4b)
    
    # Part 4c
    sizes = [100, 75, 50, 25]
    part_4c = train_different_sizes(sizes)
    p4c = PrettyTable()
    p4c.field_names = ["Train sentences", "Accuracy", "Errors (words/error)", "Test sentences"]
    column = ["news-train (100%)", "news-train (75%)", "news-train (50%)", "news-train (25%)"]
    
    for i in range(len(part_4c)):
        p4c.add_row([column[i], part_4c[i][0], part_4c[i][1], 'news-test'])
        
    print(p4c)
     
    # Part 4d
    part_4d = compare_train_test_partitions('news')
    p4d = PrettyTable()
    p4d.field_names = ["Genre", "Accuracy", "Errors (words/error)", "Partition"]
    column = ["test = news[:500], train = news[500:]", "test = news[-500:], train = news[:-500]"]
    
    for i in range(len(part_4d)):
        p4d.add_row(['news', part_4d[i][0], part_4d[i][1], column[i]])
        
    print(p4d)
    
    # Part 4e
    part_4e = compare_different_tagsets('news')
    tags = n_tags('news')
    p4e = PrettyTable()
    p4e.field_names = ["Tagset", "Accuracy", "Errors (words/error)", "Nr.tags"]
    tagset = ["news universal", "news super-simple", "news original"]
    
    for i in range(len(part_4e)):
        p4e.add_row([tagset[i], part_4e[i][0], part_4e[i][1], tags[i]])
        
    print(p4e)


if __name__ == "__main__":
    print('\nPART 1')
    part1()
    print('\nPART 2')
    part2()
    print('\nPART 3')
    part3()
    print('\nPART 4')
    part4()
