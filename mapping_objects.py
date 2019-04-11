import pickle
import json
import editdistance
import re;

from poleval.lib.entity.definitions import categories_dict

dir = '/home/szymon/juno/challenge/poleval/'


def levenshtein(s, t):
    return editdistance.eval(s, t)


def add_to_map(__map, __disambiguation, __disambiguation_helper, __entity, __tuple, __root_type):
    if not __entity:
        return
    __entity = strip_string(__entity)
    if __entity in __map.keys():
        if __entity not in __disambiguation:
            __disambiguation[__entity] = [__disambiguation_helper[__entity]]
            del (__disambiguation_helper[__entity])
        if __tuple not in __disambiguation[__entity]:
            __disambiguation[__entity].append(__tuple)
    else:
        __map[__entity] = __tuple
        __disambiguation_helper[__entity] = __tuple


def get_pickled(filename):
    with open(dir + filename + '.pickle', 'rb') as handle:
        data = pickle.load(handle)
        handle.close()
        return data


def read_json_file(json_file):
    with open(json_file) as f:
        _data = [json.loads(line) for line in f]
    f.close()
    return _data


class EntityTuple(object):
    def __init__(self, entity_en, entity_pl, original_type_id, root_type_id, pl_wiki, cleaned_pl_wiki):
        self.original_entity = pl_wiki
        self.cleaned_original_entity_space = strip_string_space(pl_wiki)
        self.original_type_id = original_type_id
        self.cleaned_original_entity = cleaned_pl_wiki
        self.root_type_id = root_type_id
        self.entity_en = entity_en
        self.entity_pl = entity_pl
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


def strip_string(_entity):
    if _entity:
        pattern = re.compile('[\W_]+')
        return pattern.sub('', _entity).lower()


def strip_string_space(_entity):
    pattern1 = re.compile('[\s]+')
    pattern2 = re.compile('[\W_]+')
    pattern3 = re.compile('XQX')
    return pattern3.sub(' ', pattern2.sub('', pattern1.sub('XQX', _entity))).lower()


def get_prefix_map(_entity_tuples):
    _prefix_map = {}
    cnt = 0
    for _e in _entity_tuples:
        if len(_e.cleaned_original_entity) > 0 and not _e.cleaned_original_entity[0] in _prefix_map:
            _prefix_map[_e.cleaned_original_entity[0]] = cnt
        cnt += 1
    return _prefix_map


def save_to_file(filename, obj):
    with open(dir + filename + '.pickle', 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()


def get_mapping_classes(jsons, types):
    _category_map = {}
    _tuples = []
    _pl_map = {}
    _en_map = {}
    _disambiguation = {}
    _disambiguation_helper = {}
    for _json in jsons:
        root_type = ''
        if 'P31' in _json.keys():
            for subtype in _json['P31']:
                if subtype in types:
                    _category_map[_json['id']] = subtype
                    root_type = subtype
        elif 'P279' in _json.keys():
            for subtype in _json['P279']:
                if subtype in types:
                    _category_map[_json['id']] = subtype
                    root_type = subtype

        if _json['wiki']['pl'] and len(_json['wiki']['pl']) > 1:
            __cleaned_entity = strip_string(_json['wiki']['pl'])
            _label_pl = strip_string(_json['labels']['pl'])
            _label_en = strip_string(_json['labels']['en'])
            _tuple = EntityTuple(_label_en, _label_pl, _json['id'], root_type,
                                 _json['wiki']['pl'], __cleaned_entity)
            _tuples.append(_tuple)
            add_to_map(_pl_map, _disambiguation, _disambiguation_helper, __cleaned_entity, _tuple, root_type)
            add_to_map(_pl_map, _disambiguation, _disambiguation_helper, _label_pl, _tuple, root_type)
    return _category_map, _tuples, _pl_map, _en_map, _disambiguation



# (category_map, entity_tuples, pl_map, en_map, disambiguation, prefix_map) = get_pickled("mapping-objects_ext")
# lemma_map = get_pickled("lemma_map")
# lemma_map_ext = get_pickled("lemma_map_ext")

entity_types_file_output = 'entity-types'
type_jsons = get_pickled(entity_types_file_output)
category_map, entity_tuples, pl_map, en_map, disambiguation = get_mapping_classes(type_jsons, categories_dict)
entity_tuples.sort(key=lambda x: x.cleaned_original_entity)
prefix_map = get_prefix_map(entity_tuples)
save_to_file("mapping-objects_ext", (category_map, entity_tuples, pl_map, en_map, disambiguation, prefix_map))