import re
import time

from gensim.utils import simple_preprocess
from lib.morfologik_2_1.custom_pymorfologik import Morfologik, ListParser

syntax_split_keyword = 'syntaxsplit'


def translate_number(org_sentence):
    res = re.sub(r'\b\d+[.,!@#$%^&*()]+\d+\b', 'numnp', org_sentence)
    res = re.sub(r'([A-Za-z]+[\d@]+[\w@]*|[\d@]+[A-Za-z]+[\w@]*)', 'numwordn', res)
    res = re.sub(r'\b\d{1,n}\b', 'numnnn', res)
    # res = re.sub(r'\b\d{2}\b', 'numnn', res)
    # res = re.sub(r'\b\d{3}\b', 'numnnn', res)
    # res = re.sub(r'\b\d{4}\b', 'numnnnn', res)
    # res = re.sub(r'\b\d{5}\b', 'numnnnnn', res)  # !!!!  TODO (brak dla wiekszych liczb)

    return res


def preprocess(text, stemmer, parser):
    # text = translate_number(text)
    text = simple_preprocess(text, max_len=100, min_len=1)
    res = stem(text, stemmer, parser)
    # res = [stem(x) for x in text]

    return res


def repair(v):
    if 'mieć' in v[1]:
        key = 'mieć'
    elif 'kapitan' in v[1]:
        key = 'kapitan'
    else:
        key = next(iter(v[1]))

    return key


def stem(word, stemmer, parser):
    val_stemmed = stemmer.stem(word, parser)
    res = []

    # _not = []
    #
    # _vals = []
    # for tuple in val_stemmed:
    #     _vals.append(tuple[0])
    #
    # for w in word:
    #     if w not in _vals:
    #         _not.append(w)

    for v in val_stemmed:
        if len(v[1]) > 0:
            key = repair(v)
            value = v[1][key][0]
            res.append((key, value))
        else:
            res.append((v[0], 'NaN'))
    return res


def prepare_data(corpuse):
    res = (' ' + syntax_split_keyword + ' ').join(corpuse)

    return res


def decode_prepare_data(corpuse):
    doc = []
    res = []
    for word in corpuse:
        if word[0] == syntax_split_keyword:
            res.append(doc)
            doc = []
        else:
            doc.append(word)

    res.append(doc)
    return res


if __name__ == "__main__":
    start = time.time()
    parser = ListParser(syntaxsplit_word=syntax_split_keyword)
    stemmer = Morfologik()
    #
    # with open('../../data/temp/pl_whole_txt.pkl', 'rb') as file:
    #     corpuse = pickle.load(file)
    #
    # corpuse = prepare_data(corpuse)
    # corpuse = preprocess(corpuse)
    # corpuse = decode_prepare_data(corpuse)
    #
    # with open('../../data/temp/pl_stem_whole_txt.pkl', 'wb') as file:
    #     pickle.dump(corpuse, file)

    corpuse = ['mój korespondencja powinna być kierowana na mam poniższy adres: unesco']
    corpuse = ['wielkiej brytanii']
    corpuse = ['metabolizmie tam 123 sdf', 'skąd ludzie', ', , )', 'Rząd odrzucił ich żądania.']
    corpuse = prepare_data(corpuse)

    corpuse = preprocess(corpuse)
    corpuse = decode_prepare_data(corpuse)
    print(corpuse)
    # with open('../../data/temp/pl_whole_txt.pkl', 'rb') as file:
    #     pl_corp = pickle.load(file)

    # with Pool(46) as p:
    #     stemmed_corp = list(p.map(preprocess, pl_corp, chunksize=2048))
    #
    # with open('../../data/temp/pl_stem_whole_txt.pkl', 'wb') as file:
    #     pickle.dump(stemmed_corp, file)

    print("execution polish data processing took: " + "{:8.4f}".format(time.time() - start) + " for " + str(
        len(corpuse)) + " docs; avg time per doc = " + "{:4.6f}".format((time.time() - start) / len(corpuse)))
