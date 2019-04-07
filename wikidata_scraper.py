import requests
import pickle
from bs4 import BeautifulSoup
import sys

dir = '/home/szymon/juno/challenge/poleval/'

def get_soup(_pl_entity_wiki):
    r = requests.get("https://pl.wikipedia.org/wiki/" + _pl_entity_wiki)
    return BeautifulSoup(r._content, 'html.parser')


def get_pickled(filename):
    with open(dir + filename + '.pickle', 'rb') as handle:
        data = pickle.load(handle)
        handle.close()
        return data


def get_divs(_soup, type_id):
    r = _soup.findAll("div", {"id": type_id})
    if r:
        return r[0]
    else:
        return None


def get_text(divs):
    text = ''
    if not divs:
        return []
    content = get_divs(divs, "bodyContent")
    if content:
        ps = content.findAll("p")
        for p in ps[:4]:
            text += p.getText()

    return text


def save_to_file(filename, obj):
    with open(dir + filename + '.pickle', 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()


def get_subclasses(divs_subclasses):
    subclasses = []
    for div_instance in divs_subclasses:
        subclasses.append(div_instance.findAll("a")[0].attrs['title'])
    return subclasses


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

