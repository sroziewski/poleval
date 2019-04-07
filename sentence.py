import csv
import pickle
import unicodedata
import pandas as pd  # For data handling
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser

import re  # For preprocessing

from time import time  # To time our operations
from collections import defaultdict
import spacy  # For preprocessing
import logging  # Setting up the loggings to monitor gensim

logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt='%H:%M:%S', level=logging.INFO)

dir = '/home/szymon/juno/challenge/poleval/'


class DataItem(object):
    def __init__(self, array):
        self.doc_id = array[0]
        self.token = array[1]
        self.lemma = array[2]
        self.preceding_space = array[3]
        self.morphosyntactic_tags = array[4]
        self.link_title = array[5]
        self.entity_id = array[6]


class Page(object):
    def __init__(self, array):
        self.page_id = array[0]
        self.title = array[1]
        self.type = array[2]


class ArticleParent(object):
    def __init__(self, array):
        self.article_id = array[0]
        self.categories = array[1:]


class CategoryParent(object):
    def __init__(self, array):
        self.category_id = array[0]
        self.parent_categories = array[1:]


class ChildArticle(object):
    def __init__(self, array):
        self.category_id = array[0]
        self.articles = array[1:]


class ChildCategory(object):
    def __init__(self, array):
        self.category_id = array[0]
        self.child_categories = array[1:]


class LinkBySource(object):
    def __init__(self, array):
        self.source_id = array[0]
        self.article_names = array[1:]


class WordTuple(object):
    def __init__(self, lemma, type):
        self.lemma = lemma
        self.type = type


def page_object_map(input_file, output_file):
    input_data_map = {}
    with open(input_file) as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        for line in reader:
            if len(line) > 0:
                page = Page(line)
                input_data_map[page.page_id] = page
        csv_file.close()
        save_to_file(output_file, input_data_map)


def article_parent_object_map(input_file, output_file):
    input_data_map = {}
    with open(input_file) as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        for line in reader:
            if len(line) > 0:
                article_parent = ArticleParent(line)
                input_data_map[article_parent.article_id] = article_parent
        csv_file.close()
        save_to_file(output_file, input_data_map)


def category_parent_object_map(input_file, output_file):
    input_data_map = {}
    with open(input_file) as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        for line in reader:
            if len(line) > 0:
                category_parent = CategoryParent(line)
                input_data_map[category_parent.category_id] = category_parent
        csv_file.close()
        save_to_file(output_file, input_data_map)


def child_article_object_map(input_file, output_file):
    input_data_map = {}
    with open(input_file) as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        for line in reader:
            if len(line) > 0:
                child_article = ChildArticle(line)
                input_data_map[child_article.category_id] = child_article
        csv_file.close()
        save_to_file(output_file, input_data_map)


def child_category_object_map(input_file, output_file):
    input_data_map = {}
    with open(input_file) as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        for line in reader:
            if len(line) > 0:
                child_article = ChildCategory(line)
                input_data_map[child_article.category_id] = child_article
        csv_file.close()
        save_to_file(output_file, input_data_map)


def link_by_source_object_map(input_file, output_file):
    input_data_map = {}
    with open(input_file) as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        for line in reader:
            if len(line) > 0:
                link_by_source = LinkBySource(line)
                input_data_map[link_by_source.source_id] = link_by_source
        csv_file.close()
        save_to_file(output_file, input_data_map)


def data_object_map(input_file, output_file_prefix):
    docs_counter = 0
    multi_counter = 0
    input_data_map = {}
    docs_limit = 10000
    with open(input_file) as tsvfile:
        tsv_reader = csv.reader(tsvfile, delimiter="\t")
        for line in tsv_reader:
            if docs_counter == docs_limit:
                save_to_file(output_file_prefix.format(multi_counter), input_data_map)
                input_data_map = {}
                multi_counter += 1

            if len(line) > 0:
                data = DataItem(line)
                if data.doc_id in input_data_map:
                    input_data_map[data.doc_id].append(data)
                else:
                    input_data_map[data.doc_id] = [data]
            docs_counter = len(input_data_map)
        save_to_file(output_file_prefix.format(multi_counter), input_data_map)
    tsvfile.close()


def save_to_file(filename, obj):
    with open(dir + filename + '.pickle', 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()


def get_pickled(filename):
    with open(dir + filename + '.pickle', 'rb') as handle:
        data = pickle.load(handle)
        handle.close()
        return data


def filter_empty_docs(docs):
    data_map = {}
    for key, doc_list in docs.items():
        data_map[key] = list(filter(lambda d: len(d.token) > 1, doc_list))
    return data_map


def filter_longer_tokens(docs):
    data_map = {}
    for key, doc_list in docs.items():
        data_map[key] = list(filter(lambda d: len(d.token.split(' ')) > 1, doc_list))
    return data_map


def get_text(docs):
    return ' '.join(map(lambda d: d.token, docs))


def get_word_tuples(docs):
    tuples = []
    for doc in docs:
        value = doc.lemma
        if len(doc.link_title) > len(doc.lemma):
            value = doc.link_title
        tuples.append(WordTuple(value, doc.morphosyntactic_tags))
    return tuples
    # return [*map(lambda d: WordTuple(d.lemma, d.morphosyntactic_tags), docs)]


def get_mentions(docs):
    return '+'.join(map(lambda x: x.link_title, filter(lambda d: len(d.link_title) > 1, docs)))


def map_docs_to_sentences(docs):
    data_map = {}
    stopwords = get_polish_stopwords()
    for key, doc_list in docs.items():
        txt = get_clean_text(get_sentences(get_word_tuples(doc_list)), stopwords)
        if len(txt) > 0:
            data_map[key] = txt
    return data_map


def cleaning(doc_list, stopwords):
    # Lemmatizes and removes stopwords
    # doc needs to be a spacy Doc object
    txt = [token.strip() for token in doc_list if not token.strip() in stopwords]
    # Word2Vec uses context words to learn the vector representation of a target word,
    # if a sentence is only one or two words long,
    # the benefit for the training is very small
    if len(txt) > 2:
        return txt


def get_clean_text(token_lists, stopwords):
    return [x for x in [cleaning(tokens, stopwords) for tokens in token_lists] if x is not None]


def get_polish_stopwords():
    with open("/juno/challenge/poleval/polish.stopwords.txt") as file:
        return [line.strip() for line in file]


def get_sentences(tuples):
    sentences = []
    handy = []
    for _ in tuples:
        if _.type == 'interp' and _.lemma == '.':
            sentences.append(handy)
            handy = []
        elif _.type != 'interp':
            handy.append(_.lemma.split(':')[0].lower().translate(
                {ord(i): None for i in ['”', '„', '(', ')', '{', '}', '[', ']']}))
    if handy:
        sentences.append(handy)
    return sentences


def map_docs_to_mentions(docs):
    data_map = {}
    for key, doc_list in docs.items():
        data_map[key] = get_mentions(doc_list)
    return data_map


def get_list_sentences(a_map):
    return [sentence for key, sentence in a_map.items()]


def get_bigram_transformer(sentences):
    # for sentence in sentences:
    #     Phrases([row.split() for row in sentences], min_count=30, progress_per=10000)
    return Phrases([sentence for sentence in sentences], min_count=30, progress_per=10000)


# input_file = dir + 'tokens-with-entities.tsv'
# input_file = dir + 'tokens-with-entities-and-tags_1mln.tsv'
saved_data_file = "100000/tokens-with-entities_{}"
# saved_data_file = "20000/tokens-with-entities-and-tags_1mln"
# data_object_map(input_file, saved_data_file)

pages_input_file = dir + 'wikipedia-data/page.csv'
article_parents_input_file = dir + 'wikipedia-data/articleParents.csv'
category_parents_input_file = dir + 'wikipedia-data/categoryParents.csv'
child_articles_input_file = dir + 'wikipedia-data/childArticles.csv'
child_categories_input_file = dir + 'wikipedia-data/childCategories.csv'
link_by_source_input_file = dir + 'wikipedia-data/linkBySource.csv'
pages_output_file = 'pages'
article_parents_output_file = 'articleParents'
category_parents_output_file = 'categoryParents'
child_articles_output_file = 'childArticles'
child_categories_output_file = 'childCategories'
link_by_source_output_file = 'linkBySource'


# page_object_map(pages_input_file, pages_output_file)
# article_parent_object_map(article_parents_input_file, article_parents_output_file)
# category_parent_object_map(category_parents_input_file, category_parents_output_file)
# child_article_object_map(child_articles_input_file, child_articles_output_file)
# child_category_object_map(child_categories_input_file, child_categories_output_file)
# link_by_source_object_map(link_by_source_input_file, link_by_source_output_file)

# saved_data_file = "1mln_tokens"


# save_to_file(saved_data_file, data)

def flatten_list(list):
    return [item for sublist in list for item in sublist]


def get_lemma_map(data):
    _lemma_map = {}
    for _data_item_list in data.values():
        for _data_item in _data_item_list:
            to_exclude = [' ', '”', '„', '(', ')', '{', '}', '[', ']', '.', ',', '!', '?', '%', '\'', '\\', '/', '+',
                          '-', '=',
                          ':', '*', '@', '#', '$', '^', '&', '_', '<', '>', '"', '÷']
            _entity = _data_item.token.lower().translate(
                {ord(i): None for i in to_exclude}) if _data_item.token else False
            _lemma_map[_entity] = _data_item.lemma
    return _lemma_map

def process_batches_for_lemma():
    for i in range(0, 19):
        data = get_pickled(saved_data_file.format(i))
        lemma_map = get_lemma_map(data)
    save_to_file("lemma_map", lemma_map)

process_batches_for_lemma()

def process_batches(sentences_out):
    for i in range(0, 19):
        data = get_pickled(saved_data_file.format(i))
        data_map_of_sentences = map_docs_to_sentences(data)
        sentences_for_docs = get_list_sentences(data_map_of_sentences)
        sentences = flatten_list(sentences_for_docs)
        sentences_out.append(sentences)


sentences_pred = []
process_batches(sentences_pred)
sentences = flatten_list(sentences_pred)

save_to_file("all-sentences", sentences)
