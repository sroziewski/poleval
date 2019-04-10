import sys

from poleval.lib.poleval import get_pickled, get_soup, get_text, save_to_file

entity_types_file_output = 'entity-types'

type_jsons = get_pickled(entity_types_file_output)

text_mapping = {}

__disambiguation = {}
__disambiguation_helper = {}

cnt = 0
multi = int(sys.argv[1])
ply = 100000


# (text_mapping, __disambiguation) = get_pickled("data-scraped-{}".format(multi))

cnt = ply * (multi - 1)
for json in type_jsons[(multi - 1) * ply:multi * ply]:
    cnt += 1
    if json['wiki']['pl']:
        to_exclude = [' ']
        _entity = json['wiki']['pl'].lower().translate(
            {ord(i): '_' for i in to_exclude}) if json['wiki']['pl'] else False
        text = get_text(get_soup(_entity))
        cnt += 1
        print("Progress {}".format(cnt)) if cnt % ply/10 == 0 else False

        if _entity in text_mapping.keys():
            if _entity not in __disambiguation:
                __disambiguation[_entity] = [__disambiguation_helper[_entity]]
                del (__disambiguation_helper[_entity])
            __disambiguation[_entity].append(text)
        else:
            text_mapping[_entity] = text
            __disambiguation_helper[_entity] = text
    if cnt % ply / 10 == 0:
        save_to_file("data-scraped-{}-{}".format(multi, cnt), (text_mapping, __disambiguation))
        text_mapping = {}

