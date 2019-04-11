# coding=utf-8

from parser import decode_prepare_data, prepare_data, preprocess
from lib.morfologik_2_1.custom_pymorfologik import Morfologik, ListParser
from gensim.models.phrases import Phrases

from time import time  # To time our operations
import logging

from poleval.lib.entity.definitions import saved_data_file_tokens_entities_tags
from poleval.lib.poleval import get_pickled, map_docs_to_sentences, get_list_sentences, get_polish_stopwords, \
    contains_digit, flatten_list, save_to_file, strip_string

logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt='%H:%M:%S', level=logging.INFO)

syntax_split_keyword = 'syntaxsplit'
parser = ListParser(syntaxsplit_word=syntax_split_keyword)
stemmer = Morfologik()



def process_batches(sentences_out):
    for i in range(0, 19):
        print(i)
        t = time()
        data = get_pickled(saved_data_file.format(i))
        data_map_of_sentences = map_docs_to_sentences(data)
        sentences_for_docs = get_list_sentences(data_map_of_sentences)
        sentences = flatten_list(sentences_for_docs)
        sentences_out.append(sentences)
        print('Process batch: {} mins'.format(round((time() - t) / 60, 2)))


def get_lemma_map(data):
    _lemma_map = {}
    stopwords = get_polish_stopwords()
    for _data_item_list in data.values():
        for _data_item in _data_item_list:
            _entity = strip_string(_data_item.token.lower())
            if _data_item.morphosyntactic_tags != 'interp':
                _value = strip_string(_data_item.lemma.split(':')[0].lower())
                if _value not in stopwords:
                    if _entity in _lemma_map and not _value in _lemma_map[_entity]:
                        _lemma_map[_entity].append(_value)
                    else:
                        _lemma_map[_entity] = [_value]
    return _lemma_map


def _get_lemma_map(data):
    stopwords = get_polish_stopwords()
    _lemma_map = {}
    for _data_item_list in data.values():
        _filtered = list(filter(lambda y: len(strip_string(y.token).lower()) > 1 and y.token.lower() not in stopwords,
                                _data_item_list))
        _handy_map = {}
        for di in _filtered:
            _handy_map[strip_string(di.token).lower()] = di.lemma.lower()
        _tokens = list(filter(lambda y: y and len(y) > 1, map(lambda x: strip_string(x.token).lower(), _filtered)))
        _numbers = list(filter(lambda y: y and contains_digit(y), _tokens))
        for _number in _numbers:
            _lemma_map[_number] = _number
        _words = list(filter(lambda y: y and not contains_digit(y), _tokens))
        _words = list(set(_words))
        _text = [' '.join(_words)]
        _corpuse = prepare_data(_text)
        _corpuse = preprocess(_corpuse, stemmer, parser)
        _corpuse = decode_prepare_data(_corpuse)
        _tuples = []
        for _i in range(0, min(len(_words), len(_corpuse[0]))):
            if 'sup' in _corpuse[0][_i][1]:
                i = 1
            if 'sup' in _corpuse[0][_i][1] or _words[_i][0:2] == _corpuse[0][_i][0][0:2]:
                # print("{} : {}".format(_i, (_words[_i], _corpuse[0][_i])))
                _tuples.append((_words[_i], _corpuse[0][_i]))

        for _word, _lemma in _tuples:
            if _lemma[1] != 'NaN':
                _lemma_map[_word] = _lemma[0]
            else:
                _lemma_map[_word] = _handy_map[_word]
    return _lemma_map


def get_bigram_transformer(sentences):
    return Phrases([sentence for sentence in sentences], min_count=30, progress_per=10000)


# t = time()
# data_object_map(input_file, saved_data_file)
# print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))


def process_batches_for_lemma():
    for i in range(0, 19):
        t = time()
        print(saved_data_file_tokens_entities_tags.format(i))
        data = get_pickled(saved_data_file_tokens_entities_tags.format(i))
        lemma_map = get_lemma_map(data)
        save_to_file("lemma_map-{}".format(i), lemma_map)

process_batches_for_lemma()
