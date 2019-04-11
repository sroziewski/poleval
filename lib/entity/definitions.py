# coding=utf-8

dir = '/home/szymon/juno/challenge/poleval/'

categories_dict = {'Q5': "człowiek", 'Q2221906': "położenie geograficzne", 'Q11862829': "dyscyplina naukowa",
                   'Q4936952': "struktura anatomiczna", 'Q12737077': "zajęcie", 'Q29048322': "model pojazdu",
                   'Q811430': "konstrukcja",
                   'Q47461344': "utwór pisany", 'Q6999': "ciało niebieskie", 'Q11460': "odzież", 'Q16521': "takson",
                   'Q24334685': "byt mityczny", 'Q31629': "dyscyplina sportu",
                   'Q28855038': "istota nadprzyrodzona", 'Q11435': "ciecz", 'Q28108': "system polityczny",
                   'Q16334298': "zwierzę", 'Q43460564': "substancja chemiczna", 'Q732577': "publikacja",
                   'Q271669': "ukształtowanie terenu", 'Q34770': "język", 'Q2198779': "jednostka",
                   'Q20719696': "obiekt geograficzny", 'Q15621286': "dzieło artystyczne", 'Q39546': "narzędzie",
                   'Q7239': "organizm", 'Q2095': "jedzenie", 'Q7184903': "obiekt abstrakcyjny", 'Q483247': "zjawisko",
                   'Q11344': "substancja", 'Q6671777': "struktura"}

categories = ["człowiek", "położenie geograficzny", "dyscyplina naukowy", "struktura anatomiczny", "zajęcie",
              "model pojazd", "konstrukcja", "utwór pisany", "ciało niebieskie", "odzież", "takson", "byt mityczny",
              "dyscyplina sport", "istota nadprzyrodzony", "ciecz", "system polityczny", "zwierzę",
              "substancja chemiczny", "publikacja", "ukształtowanie teren", "język", "jednostka", "obiekt geograficzny",
              "dzieło artystyczny", "narzędzie", "organizm", "jedzenie", "obiekt abstrakcyjny", "zjawisko",
              "substancja", "struktura"]


input_file_tokens_entities_tags = dir + 'tokens-with-entities-and-tags_1mln.tsv'
saved_data_file_tokens_entities_tags = "20000/tokens-with-entities-and-tags_1mln"
json_file = dir + 'entities.jsonl'
entity_types_file = dir + 'entity-types.tsv'
entity_types_file_output = 'entity-types'

saved_data_file_tokens_entities_tags_20000 = "20000/tokens-with-entities_{}"

input_file_test_set = dir + 'test/task3_test.tsv'
saved_data_file_test_set = "test/task3_test"

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
