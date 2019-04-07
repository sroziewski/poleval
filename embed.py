import editdistance
import pickle
import re
from gensim.models import Word2Vec
import numpy as np

dir = '/home/szymon/juno/challenge/poleval/'


def levenshtein(s, t):
    return editdistance.eval(s, t)


class EntityTuple(object):
    def __init__(self, entity_en, entity_pl, original_type_id, root_type_id, pl_wiki, cleaned_pl_wiki):
        self.original_entity = pl_wiki
        self.cleaned_original_entity_space = strip_string_space(pl_wiki)
        self.original_type_id = original_type_id
        self.cleaned_original_entity = cleaned_pl_wiki
        self.root_type_id = root_type_id
        self.entity_en = strip_string(entity_en)
        self.entity_pl = strip_string(entity_pl)
        self.set_lev_sensitivity()
        self.set_lev_similarity()

    def set_lev_similarity(self):
        if self.cleaned_original_entity:
            if len(self.cleaned_original_entity) <= 5:
                self.lev_similarity = 3
            if len(self.cleaned_original_entity) > 5:
                self.lev_similarity = 4

    def set_lev_sensitivity(self):
        if self.cleaned_original_entity:
            if len(self.cleaned_original_entity) <= 3:
                self.lev_sensitivity = 0
            if len(self.cleaned_original_entity) > 3 and len(self.cleaned_original_entity) < 6:
                self.lev_sensitivity = 2
            if len(self.cleaned_original_entity) >= 6 and len(self.cleaned_original_entity) < 10:
                self.lev_sensitivity = 3
            if len(self.cleaned_original_entity) >= 10 and len(self.cleaned_original_entity) < 14:
                self.lev_sensitivity = 4
            if len(self.cleaned_original_entity) >= 14 and len(self.cleaned_original_entity) < 18:
                self.lev_sensitivity = 5
            if len(self.cleaned_original_entity) >= 18 and len(self.cleaned_original_entity) < 22:
                self.lev_sensitivity = 6
            if len(self.cleaned_original_entity) >= 22 and len(self.cleaned_original_entity) < 26:
                self.lev_sensitivity = 7
            if len(self.cleaned_original_entity) >= 26 and len(self.cleaned_original_entity) < 30:
                self.lev_sensitivity = 8
            if len(self.cleaned_original_entity) >= 30 and len(self.cleaned_original_entity) < 34:
                self.lev_sensitivity = 9
            if len(self.cleaned_original_entity) >= 34:
                self.lev_sensitivity = 10

        elif self.entity_en:
            if len(self.entity_en) > 3 and len(self.entity_en) < 5:
                self.lev_sensitivity = 1
            if len(self.entity_en) >= 5 and len(self.entity_en) < 7:
                self.lev_sensitivity = 2
            if len(self.entity_en) >= 7 and len(self.entity_en) < 11:
                self.lev_sensitivity = 3
            if len(self.entity_en) >= 11 and len(self.entity_en) < 14:
                self.lev_sensitivity = 4
            if len(self.entity_en) >= 14 and len(self.entity_en) < 18:
                self.lev_sensitivity = 5
            if len(self.entity_en) >= 18 and len(self.entity_en) < 22:
                self.lev_sensitivity = 6
            if len(self.entity_en) >= 22 and len(self.entity_en) < 26:
                self.lev_sensitivity = 7
            if len(self.entity_en) >= 26 and len(self.entity_en) < 30:
                self.lev_sensitivity = 8
            if len(self.entity_en) >= 30 and len(self.entity_en) < 34:
                self.lev_sensitivity = 9
            if len(self.entity_en) >= 34:
                self.lev_sensitivity = 10

    def similar_to(self, _entity, _entities):
        # to_exclude = [' ', '”', '„', '(', ')', '{', '}', '[', ']', '.', ',']
        # _entity = _entity.lower().translate(
        #     {ord(i): None for i in to_exclude}) if _entity else False
        _l = self.cleaned_original_entity_space.split()
        if len(_l) > 2 and len(_entities) > 2 and _l[0][:self.lev_similarity] == _entities[0][:self.lev_similarity] and \
                _l[2][:self.lev_similarity] == _entities[2][:self.lev_similarity] and _l[1][
                                                                                      :self.lev_similarity] == \
                _entities[1][:self.lev_similarity] and self.lev_sensitivity and levenshtein(
            self.cleaned_original_entity, _entity) < self.lev_sensitivity:
            return True
        if len(_l) > 1 and len(_entities) > 1 and _l[0][:self.lev_similarity] == _entities[0][:self.lev_similarity] and \
                _l[1][
                :self.lev_similarity] == \
                _entities[1][:self.lev_similarity] and self.lev_sensitivity and levenshtein(
            self.cleaned_original_entity, _entity) < self.lev_sensitivity:
            return True

        if _l[0][:self.lev_similarity] == _entities[0][:self.lev_similarity] and self.lev_sensitivity and levenshtein(
                self.cleaned_original_entity, _entity) < self.lev_sensitivity:
            return True


class DataItem(object):
    def __init__(self, array):
        self.doc_id = array[0]
        self.token = array[1]
        self.lemma = array[2].strip()
        self.preceding_space = array[3]
        self.morphosyntactic_tags = array[4]
        self.link_title = array[5]
        self.entity_id = array[6]


def get_pickled(filename):
    with open(dir + filename + '.pickle', 'rb') as handle:
        data = pickle.load(handle)
        handle.close()
        return data


def cleaning(doc_list, stopwords):
    # Lemmatizes and removes stopwords
    # doc needs to be a spacy Doc object
    txt = [token for token in doc_list if not token.word in stopwords]
    # Word2Vec uses context words to learn the vector representation of a target word,
    # if a sentence is only one or two words long,
    # the benefit for the training is very small
    if len(txt) > 0:
        return txt


class Word(object):
    def __init__(self, word, pos, entity_id):
        self.word = word
        self.pos = pos
        self.entity_id = entity_id


class WordTriplet(object):
    def __init__(self, lemma, morpho_type, entity_id):
        self.lemma = lemma
        self.moprho_type = morpho_type
        self.entity_id = entity_id


class Entity(object):
    def __init__(self, entity, context):
        self.entity = entity
        self.context = context

    def add_context(self, _c):
        self.context.extend(_c)


def get_word_tuples(docs):
    tuples = []
    for doc in docs:
        value = doc.lemma
        if len(doc.link_title) > len(doc.lemma):
            value = doc.link_title
        _id = doc.entity_id if doc.entity_id != '_' else ''
        tuples.append(WordTriplet(value, doc.morphosyntactic_tags, _id))
    return tuples


def get_polish_stopwords():
    with open(dir + "polish.stopwords.txt") as file:
        return [line.strip() for line in file]


def get_clean_text(token_lists, stopwords):
    return [x for x in [cleaning(tokens, stopwords) for tokens in token_lists] if x is not None]


def strip_string(_entity):
    if _entity:
        pattern = re.compile('[\W_]+')
        return pattern.sub('', _entity).lower()


def get_sentences(tuples):
    sentences = []
    handy = []
    for _ in tuples:
        if _.moprho_type == 'interp:' and _.lemma == '.':
            sentences.append(handy)
            handy = []
        elif _.moprho_type != 'interp:':
            handy.append(Word(strip_string(_.lemma.split(':')[0]), _.moprho_type, _.entity_id))
    if handy:
        sentences.append(handy)
    return sentences


def contains_entity(_words):
    _ent = []
    for _word in _words:
        if _word.entity_id:
            _ent.append(_word)
    return _ent if len(_ent) > 0 else False


def save_to_file(filename, obj):
    with open(dir + filename + '.pickle', 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()


def extract_context(_sentence, _entity, __context):
    _l = []
    for _word in _sentence:
        if _word.pos.split(":")[0] == _entity.pos.split(":")[0]:
            _l.append(_word) if not _word.entity_id == _entity.entity_id and _word.word not in __context else None
    return _l


def flatten_list(_list):
    return [item for sublist in _list for item in sublist]


def add_to_list(_sentence_list, _sentences):
    for _s in _sentences:
        _sentence_list.append(_s)


def list_docs_to_sentences(docs):
    _sentence_list = []
    stopwords = get_polish_stopwords()
    for key, doc_list in docs.items():
        _sentences = get_clean_text(get_sentences(get_word_tuples(doc_list)), stopwords)
        if len(_sentences) > 0:
            add_to_list(_sentence_list, _sentences)
    return _sentence_list


def get_test_data(data):
    _tuples = []
    for _data_item_list in data.values():
        _test_tuples = {}
        for _data_item in _data_item_list:
            if _data_item.entity_id != '_':
                if _data_item.entity_id not in _test_tuples:
                    _test_tuples[_data_item.entity_id] = [strip_string(_data_item.token)]
                elif _data_item.token not in _test_tuples[_data_item.entity_id]:
                    _test_tuples[_data_item.entity_id].append(strip_string(_data_item.token))
        if len(_test_tuples) > 0:
            _tuples.append(_test_tuples)
    return _tuples


def manage_doc_context(_sentence_list):
    _was_entity = False
    _c = 0
    _entity_map = {}
    for _sentence in _sentence_list:
        if _was_entity:
            for _e in _was_entity:
                try:
                    _l2 = extract_context(_sentence_list[_c], _e, _entity_map[_e.entity_id].context)
                    _entity_map[_e.entity_id].add_context(_l2)
                except:
                    i = 1
        _entity_l = contains_entity(_sentence)
        if _entity_l:
            _l1 = False
            _m1 = {}
            for _e in _entity_l:
                if _c - 1 >= len(_sentence_list):
                    _l1 = extract_context(_sentence_list[_c - 1], _e, [])
                _l0 = extract_context(_sentence_list[_c], _e, [])
                if _l1:
                    _l1.extend(_l0)
                    _l = _l1
                else:
                    _l = _l0
                _m1[_e] = _l
            _was_entity = _entity_l
            for _e in _entity_l:
                if _e.entity_id not in _entity_map:
                    _entity_map[_e.entity_id] = Entity(_e, _m1[_e])
        else:
            _was_entity = False
        _c += 1
    return _entity_map


input_file = dir + 'test/task3_test.tsv'
saved_data_file = "test/task3_test"


# data_object_map(input_file, saved_data_file)
# data = get_pickled(saved_data_file)
# test_tuples = get_test_data(data)


# saved_data_file = "20000/tokens-with-entities-and-tags_1mln"

# data = get_pickled(saved_data_file)
# sentences = list_docs_to_sentences(data)
# save_to_file("sentence_list", sentences)
# manage_doc_context(list_docs_to_sentences(data))
# sentences = get_pickled("sentence_list")
# entity_map = manage_doc_context(sentences)
# save_to_file("entity_map-valid", entity_map)


def get_avg_similarity(_tuple, _context, _w2vec):
    _avg_l = []
    for _w1 in _context:
        for _w2 in _tuple[1]:
            try:
                _avg_l.append(_w2vec.wv.distance(_w1, _w2))
            except KeyError:
                pass
    return np.mean(_avg_l)


def filter_singleton_tuple(_key, _tuples):
    _n_t = []
    for __t in _tuples:
        if len(__t[1]) == 1 and list(__t[1])[0] == _key:
            pass
        else:
            _n_t.append(__t)
    return _n_t


def map_to_valid(_entity_valid_map, _found, _found_with_error, _wikidata_context_map, _w2vec, _scrap_found_map):
    _mapped_test = []
    _not_mapped = []
    _hash_map = {}
    for _k, _v in _entity_valid_map.items():
        if _v.entity.word in _wikidata_context_map:
            _tuples = _wikidata_context_map[_v.entity.word]
            _max_l = []
            _context = set(list(map(lambda x: x.word, _v.context)))
            __clean_tuples = filter_singleton_tuple(_v.entity.word, _tuples)
            if len(__clean_tuples) == 0:
                __strip_Q_in_tuple_name = []
                for _x in _tuples:
                    __strip_Q_in_tuple_name.append(int(_x[0][1:]))
                __strip_Q_found = 'Q{}'.format(min(__strip_Q_in_tuple_name))
                _mapped_test.append((_v.entity.entity_id, __strip_Q_found))
                _hash_map[_v.entity.word] = __strip_Q_found
            else:
                for _t in __clean_tuples:
                    _max_l.append((_t, get_avg_similarity(_t, _context, _w2vec)))

                _found_tuple = min(_max_l, key=lambda _item: _item[1])
                _mapped_test.append((_v.entity.entity_id, _found_tuple[0][0]))
                _hash_map[_v.entity.word] = _found_tuple[0][0]
            i = 1
        elif _v.entity.word in _found:
            _mapped_test.append((_v.entity.entity_id, _found[_v.entity.word][0]))
            _hash_map[_v.entity.word] = _found[_v.entity.word][0]
        elif _v.entity.word in _found_with_error:
            _mapped_test.append((_v.entity.entity_id, _found_with_error[_v.entity.word][0]))
            _hash_map[_v.entity.word] = _found_with_error[_v.entity.word][0]
        else:
            if _v.entity.entity_id in _scrap_found_map:
                _mapped_test.append((_v.entity.entity_id, _scrap_found_map[_v.entity.entity_id]))
                _hash_map[_v.entity.word] = _scrap_found_map[_v.entity.entity_id]
            else:
                if _v.entity.word in _hash_map:
                    _mapped_test.append((_v.entity.entity_id, _hash_map[_v.entity.word]))
                else:
                    _mapped_test.append((_v.entity.entity_id, ''))
                    _not_mapped.append((_k, _v))

    return _mapped_test, _not_mapped


def filter_tuples(_word, _clean_tuples_):
    _filtered = []
    for _t_ in _clean_tuples_:
        _has = False
        for _w_ in _t_[1]:
            if _w_[0:3] == _word[0:3]:
                _has = True
        if _has:
            _filtered.append(_t_)
    return _filtered


def filter_found(_tuples):
    _filtered_found = []
    for ___t in _tuples:
        if len(___t[0][0]) < 7:
            _filtered_found.append(___t)
    return _filtered_found


def map_scrapped_not_found(_tuples, _w2vec):
    _scrapped_mapped_test = []
    _scrapped_not_mapped_test = []

    for _t in _tuples:
        if _t[0] == 'e1753':
            jj = 1
        _max_l = []
        _context = set(list(map(lambda x: x.word, _t[1].context)))
        if len(_t[2]) == 0:
            _scrapped_not_mapped_test.append((_t[1].entity.entity_id, _t[1].entity.word))
        else:
            __clean_tuples = filter_singleton_tuple(_t[1].entity.word, _t[2])
            __filtered_tuples = filter_tuples(_t[1].entity.word, __clean_tuples)
            if len(__filtered_tuples) == 0:
                __strip_Q_in_tuple_name = []
                for _x in _t[2]:
                    __strip_Q_in_tuple_name.append(int(_x[0][1:]))
                __strip_Q_found = 'Q{}'.format(min(__strip_Q_in_tuple_name))
                _scrapped_mapped_test.append((_t[1].entity.entity_id, __strip_Q_found))
            else:
                for __t in __filtered_tuples:
                    _max_l.append((__t, get_avg_similarity(__t, _context, _w2vec)))

                _max_l_f = filter_found(_max_l)
                if len(_max_l_f) > 0:
                    _found_tuple = min(_max_l_f, key=lambda _item: _item[1])
                else:
                    _found_tuple = min(_max_l, key=lambda _item: _item[1])
                _scrapped_mapped_test.append((_t[1].entity.entity_id, _found_tuple[0][0]))

    return _scrapped_mapped_test, _scrapped_not_mapped_test


def filter_found_with_error(_found_with_error):
    _filtered = {}
    for _k, _v in _found_with_error.items():
        if len(_v) == 1:
            _filtered[_k] = _v
    return _filtered


def filter_found_with_error_more(_found_with_error):
    _filtered = {}
    for _k, _v in _found_with_error.items():
        if len(_v) > 1:
            _filtered[_k] = _v
    return _filtered


def try_to_find(_not_found_errors_list_of_tuples_chunk, _entity):
    _i = 0
    _candidates = {}
    for _et in _not_found_errors_list_of_tuples_chunk:
        try:
            if _et.original_type_id not in _candidates and levenshtein(_et.cleaned_original_entity, _entity) < _et.lev_sensitivity - 1:
                _candidates[_et.original_type_id] = _et
                if len(_candidates) > 3:
                    return _candidates
        except:
            AttributeError
        _i += 1
    return _candidates


def clean_tuples(_list):
    _new_list2 = []
    for _tuple in _list:
        _e1 = extract_tuple_text(_tuple[1])
        _e1.extend(extract_tuple_text(_tuple[2])) if _tuple[2] else None
        _new_list2.append((_tuple[0], set(flatten_list(_e1))))
    return _new_list2


def get_entity(_e):
    _t = strip_string(_e).lower()
    if _t and _t not in stopwords:
        return lemma_map[_t] if not _e.isdigit() and _t in lemma_map else [_t]


def extract_tuple_text(_text):
    _l1 = []
    for _t in _text.split():
        if _t:
            # _ee = get_entity(_t, lemma_map, stopwords)
            _ee = get_entity(_t)
            if _ee:
                _l1.append(_ee)
    return _l1


def map_scrap_found(_scrap_found_and_not_found):
    _scrap_found_map = {}
    for _t in _scrap_found_and_not_found:
        _scrap_found_map[_t[0]] = _t[1]
    return _scrap_found_map


stopwords = get_polish_stopwords()
lemma_map = get_pickled("lemma_map_ext")

found_with_error = get_pickled("merged_map_found_with_error")
found = get_pickled("merged_map_found")
# (category_map, entity_tuples, pl_map, en_map, disambiguation, prefix_map) = get_pickled("mapping-objects_ext")
wikidata_context_map = get_pickled("wikidata_context_map")
entity_valid_map = get_pickled("entity_map-valid")

scrap_found_map = get_pickled("scrap_found_map")

filtered_found_with_error = filter_found_with_error(found_with_error)
filtered_found_with_error_more = filter_found_with_error_more(found_with_error)

w2vec = Word2Vec.load(dir + "all-sentences-word2vec-m3.model")

# mapped, not_mapped = map_to_valid(entity_valid_map, found, filtered_found_with_error, wikidata_context_map, w2vec, scrap_found_map)
# save_to_file("map_test_set_with_scrap_filled", (mapped, not_mapped))

(mapped, not_mapped) = get_pickled("map_test_set_with_scrap_filled")

import csv
with open(dir+'roziewski-poleval-task3.tsv', 'w') as tsvfile:
    writer = csv.writer(tsvfile, delimiter='=')
    for _t in mapped:
        writer.writerow([_t[0], _t[1]])

(mapped, not_mapped) = get_pickled("map_test_set_with_scrap")
# (mapped, not_mapped) = get_pickled("map_test_set")


clean_scrapped_not_found = get_pickled("clean_scrapped_not_found")
scrap_mapped, scrap_not_mapped = get_pickled("scrap_found_and_not_found")

scrap_found_map = map_scrap_found(scrap_mapped)
save_to_file("scrap_found_map", scrap_found_map)

# scrap_mapped, scrap_not_mapped = map_scrapped_not_found(clean_scrapped_not_found, w2vec)
# save_to_file("scrap_found_and_not_found", (scrap_mapped, scrap_not_mapped))

i = 1
# clean_scrapped_not_found = []
# for _t in scrapped_not_found:
#     _clean_tuple = (_t[0], _t[1], clean_tuples(_t[2]))
#     clean_scrapped_not_found.append(_clean_tuple)
#
# save_to_file("clean_scrapped_not_found", clean_scrapped_not_found)

# not_found_errors_list_of_tuples_chunk1 = get_pickled("not_mapped_found_candidates-5")

# _to_scrap = []
# for _t in not_found_errors_list_of_tuples_chunk1:
#     _to_scrap.append((_t[0][0], _t[0][1], list(_t[1].keys())))


# not_found_errors_list_of_tuples = get_pickled("not_found_errors_list_of_tuples")


# nf = get_pickled("not_mapped_found_candidates-2")
#
#
# scrapped_not_found = get_pickled("scrapped_not_found-{}".format(1))
# for i in range(2, 8):
#     _scrapped_not_found = get_pickled("scrapped_not_found-{}".format(i))
#     scrapped_not_found.extend(_scrapped_not_found)

multi = 1

# not_mapped_found_candidates = []
#
# k = 0
# for _t in not_mapped[0:50]:
#     print("script: {} , {}".format(multi, k))
#     _f = try_to_find(not_found_errors_list_of_tuples_chunk1, _t[1].entity.word)
#     for __k, __v in _f.items():
#         print("{} : {}".format(__k, _t[1].entity.word))
#     not_mapped_found_candidates.append((_t, _f))
#     k += 1
#
# save_to_file("not_mapped_found_candidates-{}".format(multi), not_mapped_found_candidates)
#
#
# f = try_to_find(not_found_errors_list_of_tuples_chunk1, 'wałęsa')


i = 1
