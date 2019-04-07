import pickle
import csv
import re
import editdistance


def levenshtein(s, t):
    return editdistance.eval(s, t)


def strip_string(_entity):
    pattern = re.compile('[\W_]+')
    return pattern.sub('', _entity)


def strip_string_space(_entity):
    to_exclude = ['”', '„', '(', ')', '{', '}', '[', ']', '.', ',', '!', '?', '%', '\'', '\\', '/', '+', '-', '=',
                  ':', '*', '@', '#', '$', '^', '&', '_', '<', '>', '"', '÷', '’']
    _entity = _entity.lower().translate(
        {ord(i): None for i in to_exclude}) if _entity else False
    return _entity


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
                try:
                    data = DataItem(line)
                except IndexError:
                    if 'e' in line[-1]:
                        j = 1
                    i = 1
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


class DataItem(object):
    def __init__(self, array):
        self.doc_id = array[0]
        self.token = array[1]
        self.lemma = array[2]
        self.preceding_space = array[3]
        self.morphosyntactic_tags = array[4]
        self.link_title = array[5]
        self.entity_id = array[6]


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
        if len(_l) != len(_entities):
            return False
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


dir = '/home/szymon/juno/challenge/poleval/'

input_file = dir + 'test/task3_test.tsv'
saved_data_file = "test/task3_test"

data_object_map(input_file, saved_data_file)
data = get_pickled(saved_data_file)
test_tuples = get_test_data(data)

(category_map, entity_tuples, pl_map, en_map, disambiguation, prefix_map) = get_pickled("mapping-objects_ext")
lemma_map = get_pickled("lemma_map_ext")

i = 1

