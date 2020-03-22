#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 09:08:19 2020

@author: ronjaronnback
"""

import json
import numpy as np
from collections import Counter, defaultdict

train_path = '/Users/ronjaronnback/Desktop/University/Year_3/Semester 2/Computational Linguistics/Assignment/_0.json'
test_path = '/Users/ronjaronnback/Desktop/University/Year_3/Semester 2/Computational Linguistics/Assignment/_1.json'

#------------------------------------------------------------------------------
# CORPUS
#------------------------------------------------------------------------------


class Corpus(object):
    
    """
    This class creates a corpus object read off a .json file consisting of a list of lists,
    where each inner list is a sentence encoded as a list of strings.
    """
    
    def __init__(self, path, t, n, bos_eos=True, vocab=None):
        
        """
        DON'T TOUCH THIS CLASS! IT'S HERE TO SHOW THE PROCESS, YOU DON'T NEED TO ANYTHING HERE. 
        
        A Corpus object has the following attributes:
         - vocab: set or None (default). If a set is passed, words in the input .json file not 
                         found in the set are replaced with the UNK string
         - path: str, the path to the .json file used to build the corpus object
         - t: int, words with a lower frequency count than t are replaced with the UNK string
         - ngram_size: int, 2 for bigrams, 3 for trigrams, and so on.
         - bos_eos: bool, default to True. If False, bos and eos symbols are not prepended and appended to sentences.
         - sentences: list of lists, containing the input sentences after lowercasing and 
                         splitting at the white space
         - frequencies: Counter, mapping tokens to their frequency count in the corpus
        """
        
        self.vocab = vocab        
        self.path = path
        self.t = t
        self.ngram_size = n
        self.bos_eos = bos_eos
        
        # input --> [['I am home.'], ['You went to the park.'], ...]
        self.sentences = self.read()
        # output --> [['i', 'am', 'home' '.'], ['you', 'went', 'to', 'the', 'park', '.'], ...]
    
        self.frequencies = self.freq_distr()
        # output --> Counter('the': 485099, 'of': 301877, 'i': 286549, ...)
        # the numbers are made up, they aren't the actual frequency counts
        
        if self.t or self.vocab:
            # output --> [['i', 'am', 'home' '.'], ['you', 'went', 'to', 'the', 'park', '.'], ...]
            self.sentences = self.filter_words()
            # output --> [['i', 'am', 'home' '.'], ['you', 'went', 'to', 'the', 'UNK', '.'], ...]
            # supposing that park wasn't frequent enough or was outside of the training vocabulary, it gets
            # replaced by the UNK string
            
        if self.bos_eos:
            # output --> [['i', 'am', 'home' '.'], ['you', 'went', 'to', 'the', 'park', '.'], ...]
            self.sentences = self.add_bos_eos()
            # output --> [['eos', i', 'am', 'home' '.', 'bos'], ['eos', you', 'went', 'to', 'the', 'park', '.', 'bos'], ...]
                    
    def read(self):
        
        """
        Reads the sentences off the .json file, replaces quotes, lowercases strings and splits 
        at the white space. Returns a list of lists.
        """
        
        if self.path.endswith('.json'):
            sentences = json.load(open(self.path, 'r'))                
        else:   
            sentences = []
            with open(self.path, 'r', encoding='latin-1') as f:
                for line in f:
                    print(line[:20])
                    # first strip away newline symbols and the like, then replace ' and " with the empty 
                    # string and get rid of possible remaining trailing spaces 
                    line = line.strip().translate({ord(i): None for i in '"\'\\'}).strip(' ')
                    # lowercase and split at the white space (the corpus has ben previously tokenized)
                    sentences.append(line.lower().split(' '))
        
        return sentences
    
    def freq_distr(self):
        
        """
        Creates a counter mapping tokens to frequency counts
        
        count = Counter()
        for sentence in self.sentences:
            for word in sentence:
                count[w] += 1
            
        """
    
        return Counter([word for sentence in self.sentences for word in sentence])
        
    
    def filter_words(self):
        
        """
        Replaces illegal tokens with the UNK string. A token is illegal if its frequency count
        is lower than the given threshold and/or if it falls outside the specified vocabulary.
        The two filters can be both active at the same time but don't have to be. To exclude the 
        frequency filter, set t=0 in the class call.
        """
        
        filtered_sentences = []
        for sentence in self.sentences:
            filtered_sentence = []
            for word in sentence:
                if self.t and self.vocab:
                    # check that the word is frequent enough and occurs in the vocabulary
                    filtered_sentence.append(
                        word if self.frequencies[word] > self.t and word in self.vocab else 'UNK'
                    )
                else:
                    if self.t:
                        # check that the word is frequent enough
                        filtered_sentence.append(word if self.frequencies[word] > self.t else 'UNK')
                    else:
                        # check if the word occurs in the vocabulary
                        filtered_sentence.append(word if word in self.vocab else 'UNK')
                        
            if len(filtered_sentence) > 1:
                # make sure that the sentence contains more than 1 token
                filtered_sentences.append(filtered_sentence)
    
        return filtered_sentences
    
    def add_bos_eos(self): # PADDING ADDED HERE
        
        """
        Adds the necessary number of BOS symbols and one EOS symbol.
        
        In a bigram model, you need on bos and one eos; in a trigram model you need two bos and one eos, and so on...
        """
        
        padded_sentences = []
        for sentence in self.sentences:
            padded_sentence = ['#bos#']*(self.ngram_size-1) + sentence + ['#eos#']
            padded_sentences.append(padded_sentence)
    
        return padded_sentences


#------------------------------------------------------------------------------
# LANGUAGE MODEL
#------------------------------------------------------------------------------

def cache(func):
    print("I'm checking cache!")
    cache = {}

    def check_cache(*args):
        if args in cache:
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result

    return check_cache


class LM(object):
    
    """
    Creates a language model object which can be trained and tested.
    The language model has the following attributes:
     - vocab: set of strings
     - lam: float, indicating the constant to add to transition counts to smooth them (default to 1)
     - ngram_size: int, the size of the ngrams
    """
    
    def __init__(self, n, vocab=None, smooth='Laplace', lam=1):
        
        self.vocab = vocab
        self.lam = lam
        self.ngram_size = n
        self.probability = 0 # SELF ADDED
        self.unigram_probs = {}
        self.new_dict = {}
        
    def get_ngram(self, sentence, i):
        
        """
        CHANGE AT OWN RISK.
        
        Takes in a list of string and an index, and returns the history and current 
        token of the appropriate size: the current token is the one at the provided 
        index, while the history consists of the n-1 previous tokens. If the ngram 
        size is 1, only the current token is returned.
        
        Example:
        input sentence: ['bos', 'i', 'am', 'home', '']
        target index: 2
        ngram size: 3
        
        ngram = ['bos', 'i', 'am']  
        #from index 2-(3-1) = 0 to index i (the +1 is just because of how Python slices lists) 
        
        history = ('bos', 'i')
        target = 'am'
        return (('bos', 'i'), 'am')
        """
        
        if self.ngram_size == 1:
            return sentence[i]
        else:
            ngram = sentence[i-(n-1):i+1]
            history = tuple(ngram[:-1])
            target = ngram[-1]
            return (history, target)
                    
    def update_counts(self, corpus):
        
        """
        CHANGE AT OWN RISK.
        
        Creates a transition matrix with counts in the form of a default dict mapping history
        states to current states to the co-occurrence count (unless the ngram size is 1, in which
        case the transition matrix is a simple counter mapping tokens to frequencies. 
        The ngram size of the corpus object has to be the same as the language model ngram size.
        The input corpus (passed by providing the corpus object) is processed by extracting ngrams
        of the chosen size and updating transition counts.
        
        This method creates three attributes for the language model object:
         - counts: dict, described above
         - vocab: set, containing all the tokens in the corpus
         - vocab_size: int, indicating the number of tokens in the vocabulary
        """
        
        if self.ngram_size != corpus.ngram_size:
            raise ValueError("The corpus was pre-processed considering an ngram size of {} while the "
                             "language model was created with an ngram size of {}. \n"
                             "Please choose the same ngram size for pre-processing the corpus and fitting "
                             "the model.".format(corpus.ngram_size, self.ngram_size))
        
        self.counts = defaultdict(dict) if self.ngram_size > 1 else Counter()
        for sentence in corpus.sentences:
            for idx in range(self.ngram_size-1, len(sentence)):
                ngram = self.get_ngram(sentence, idx)
                if self.ngram_size == 1:
                    self.counts[ngram] += 1
                else:
                    # it's faster to try to do something and catch an exception than to use an if statement to check
                    # whether a condition is met beforehand. The if is checked everytime, the exception is only catched
                    # the first time, after that everything runs smoothly
                    try:
                        self.counts[ngram[0]][ngram[1]] += 1
                    except KeyError:
                        self.counts[ngram[0]][ngram[1]] = 1
        
        # first loop through the sentences in the corpus, than loop through each word in a sentence
        self.vocab = {word for sentence in corpus.sentences for word in sentence}
        self.vocab_size = len(self.vocab)
        
        
          
        if self.ngram_size == 1:
            tot = sum(list(self.counts.values())) + (self.vocab_size*self.lam)
            for word in self.counts:
                if self.counts[word] != 0:
                    ngram_count = self.counts[word] + self.lam
                else:
                    ngram_count = self.lam
                
                prob = ngram_count/tot
                self.unigram_probs[word] = prob
        
        
        if self.ngram_size == 2:
            
            self.new_dict = {}
            for i in self.counts.items():
                self.new_dict[i[0][0]] = np.sum(list(i[1].values()))
                
            self.compute_unigram_for_bigram()
            
      
    def compute_unigram_for_bigram(self):
        
        try:
            tot = np.sum(list(self.new_dict.values())) + (self.vocab_size*self.lam) # sum in all the
            
            
            for word in self.new_dict:
                try:
                    ngram_count = self.new_dict[word] + self.lam
                except KeyError:
                    ngram_count = self.lam
                
                prob = ngram_count/tot
                self.unigram_probs[word] = prob
        except KeyError:       
            self.unigram_probs[word] = self.lam / self.vocab_size*self.lam  
    
    
    def get_unigram_probability(self, ngram):
        
        """
        CHANGE THIS.
        
        Compute the probability of a given unigram in the estimated language model using
        Laplace smoothing (add k).¨
        
        for interpolation can add lambda here
        
        """
        
        return self.unigram_probs[ngram]
    
    def get_ngram_probability(self, history, target):
        
        """
        CHANGE THIS.
        
        Compute the conditional probability of the target token given the history, using 
        Laplace smoothing (add k).
        
        """
        
        total_probability = 0
        
        # BIGRAM - ONLY WORKING ONE FOR NOW
        try:
            bigram_tot = np.sum(list(self.counts[history].values())) + (self.vocab_size*self.lam) # sum in all the
            try:
                bigram_transition_count = self.counts[history][target] + self.lam # if history not in vocab, smooth
            except KeyError:
                bigram_transition_count = self.lam
        except KeyError: #
            bigram_transition_count = self.lam
            bigram_tot = self.vocab_size*self.lam
        
        bigram_probability = bigram_transition_count/bigram_tot
        
        
        # UNIGRAM    
        
        #kept getting weird errors with this but this fixes it....
        if target == '#eos#':
            unigram_probability = self.lam / self.vocab_size*self.lam  
        else:
            unigram_probability = self.unigram_probs[target]
        
        total_probability = unigram_probability*0.17 + bigram_probability*0.83 
        
        return total_probability
        
    
    def perplexity(self, test_corpus):
        
        """
        Uses the estimated language model to process a corpus and computes the perplexity 
        of the language model over the corpus.
        
        DON'T TOUCH THIS FUNCTION!!!
        
        probs = []
        total = len(test_corpus.sentences[:1000])
        counter = 0
        
        for sentence in test_corpus.sentences[:1000]:
            
            counter += 1
            if counter % 100 == 0:
                print("% done: ", counter/total * 100)
                print()
            
            for idx in range(self.ngram_size-1, len(sentence)):
                ngram = self.get_ngram(sentence, idx)
                if self.ngram_size == 1:
                    probs.append(self.get_unigram_probability(ngram))
                else:
                    probs.append(self.get_ngram_probability(ngram[0], ngram[1]))
        
        entropy = np.log2(probs)
        # this assertion makes sure that you retrieved valid probabilities, whose log must be <= 0
        #assert all(entropy <= 0)
        
        avg_entropy = -1 * (sum(entropy) / len(entropy))
        
        return pow(2.0, avg_entropy)
        """
        
        probs = []
        for sentence in test_corpus.sentences:
            for idx in range(self.ngram_size-1, len(sentence)):
                ngram = self.get_ngram(sentence, idx)
                if self.ngram_size == 1:
                    probs.append(self.get_unigram_probability(ngram))
                else:
                    probs.append(self.get_ngram_probability(ngram[0], ngram[1]))
        
        entropy = np.log2(probs)
        # this assertion makes sure that you retrieved valid probabilities, whose log must be <= 0
        assert all(entropy <= 0)
        
        avg_entropy = -1 * (sum(entropy) / len(entropy))
        
        return pow(2.0, avg_entropy)
 


#------------------------------------------------------------------------------
# UNIGRAM MODEL
#------------------------------------------------------------------------------

# example code to run a unigram model with add 0.001 smoothing. Tokens with a frequency count lower than 10
# are replaced with the UNK string

n = 1 # order of the model - unigram or higher
train_corpus = Corpus(train_path, 10, n, bos_eos=True, vocab=None)
unigram_model = LM(n, lam=0.001)
unigram_model.update_counts(train_corpus) # creates transition matrix, raw counts, not probabilities

# to ensure consistency, the test corpus is filtered using the vocabulary of the trained language model

test_corpus = Corpus(test_path, None, n, bos_eos=True, vocab=unigram_model.vocab) 
# None refers to frequency threshold
# We pass trained vocabulary because the two have to be consistent 
# (if word not in training set, then to match unknown word)
print(unigram_model.perplexity(test_corpus))

#------------------------------------------------------------------------------
# BIGRAM MODEL
#------------------------------------------------------------------------------

n = 2
train_corpus = Corpus(train_path, 10, n, bos_eos=True, vocab=None)

print("Starting up LM")
bigram_model = LM(n, lam=0.001)
print("Updating counts of LM")
bigram_model.update_counts(train_corpus)
print("LM update counts is DONE!")

# to ensure consistency, the test corpus is filtered using the vocabulary of the trained language model
test_corpus = Corpus(test_path, None, n, bos_eos=True, vocab=bigram_model.vocab)
print("Testing...")
print(bigram_model.perplexity(test_corpus))