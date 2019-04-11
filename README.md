# This is a solution to Poleval'19 Task number 3 related to Entity Linking


###Below the Task description



##Task 3: Entity linking
###Task definition
The task covers the identification of mentions of entities from a knowledge base (KB) in Polish texts. In this task as the reference KB we will use WikiData (WD), an offspring of Wikipedia – a knowledge base, that unifies structured data available in various editions of Wikipedia.

For instance the following text:


Zaginieni 11-latkowie w środę rano wyszli z domów do szkoły w Nowym Targu, gdzie przebywali do godziny 12:00. Jak informuje "Tygodnik Podhalański", 11-letni Ivan już się odnalazł, ale los Mariusza Gajdy wciąż jest nieznany. Chłopcy od chwili zaginięcia przebywali razem między innymi w Zakopanem. Mieli się rozstać w czwartek rano.
Source: gazeta.pl

has 3 entity mentions:

* Nowym Targu - https://www.wikidata.org/wiki/Q231593
* Tygodnik Podhalański- https://www.wikidata.org/wiki/Q9363509
* Zakopanem - https://www.wikidata.org/wiki/Q144786

Even though there are more mentions that have their corresponding entries in WD (such as “środa”, “dom”, “12:00”, etc.) we restrict the set of entities to a closed group of WD types, to names of countries, cities, people, occupations, organisms, tools, constructions, etc. (with important exclusion of times and dates). The full list of entity types is given at the end of the task description as well as available for DOWNLOAD. It should be also noted that names such as “Ivan” and “Mariusz Gajda” should not be recognized, since they lack corresponding entries in WD.

The task is similar to Named Entity Recognition (NER), with the important difference that in EL the set of entities is closed. To some extent EL is also similar to Word Sense Disambiguation (WSD), since mentions are ambiguous between competing entities.

In this task we have decided to ignore nested mentions of entities, so names such as “Zespół Szkół Łączności im. Obrońców Poczty Polskiej w Gdańsku, w Krakowie”, which has an entry in WikiData should be treated as an atomic linguistic unit, even though there are many entities that have their corresponding WikiData entries (such as Poczta Polska w Gdańsku, Gdańsk, Kraków). Also the algorithm is required to identify all mentions of the entity in the given document, even if they are exactly same as the previous mentions.

### Training data
The most common training data used in EL is Wikipedia itself. Even though it wasn’t designed as a reference corpus for that task, the structure of internal links serves as a good source for training and testing data, since the number of links inside Wikipedia is counted in millions. The important difference between the Wikipedia links and EL to WikiData is the fact that the titles of the Wikipedia articles evolve, while the WD identifiers remain constant. We will soon provide a portion of Wikipedia text with reference mapping of the titles into WD entities. Still it is fairly easy to obtain, since most of WD entries include a link to at least one Wikipedia.

The second important difference is the fact that according to the Wikipedia editing rules, a link should be provided only for the first mention of any salient concept present in an article. It is different from the requirements of the task in which all mentions have to be identified.

The following training data is available:

tokenised and sentence-split Wikipedia text (DOWNLOAD)
tokenised, sentence-split, tagged and lemmatized Wikipedia text (DOWNLOAD)
list of selected Wikidata types (DOWNLOAD)
Wikidata items (DOWNLOAD)
various data extracted from Wikipedia - the meaning of each file is provided in the readme.txt file (DOWNLOAD)
The data in the first and the second dataset have sentences separated by an empty line. Each line in the first dataset contains the following data (separated by tab character):

[doc_id, token, preceding_space, link_title, entity_id]

doc_id – an internal Wikipedia identifier of the article; it may be used to disambiguate entities collectively in a single document (by using internal coherence of entity mentions),
token – the value of the token,
preceding_space – 1 indicates that the token was preceded by a blank character (space in the most of the cases), 0 otherwise,
link_title – the title of the Wikipedia article that is a target of an internal link containing given token; some of the links point to articles that do not exist in Wikipedia; _ (underscore) is used when the token is not part of a link,
entity_id – the ID of the entity in Wikidata; this value has to be determined by the algorithm; _ (underscore) is used when the ID could not be established.

* Sample data

* 2 Nazwa 1 _ _
* 2 języka 1 _ _
* 2 pochodzi 1 _ _
* 2 od 1 _ _
* 2 pierwszych 1 _ _
* 2 liter 1 _ _
* 2 nazwisk 1 _ _
* 2 jego 1 _ _
* 2 autorów 1 _ _
* 2 Alfreda 1 Alfred V. Aho Q62898
* 2 V 1 Alfred V. Aho Q62898
* 2 . 0 Alfred V. Aho Q62898
* 2 Aho 1 Alfred V. Aho Q62898
* 2 , 0 _ _
* 2 Petera 1 Peter Weinberger _
* 2 Weinbergera 1 Peter Weinberger _
* 2 i 1 _ _
* 2 Briana 1 Brian Kernighan Q92608
* 2 Kernighana 1 Brian Kernighan Q92608
* 2 i 1 _ _
* 2 czasami 1 _ _
* 2 jest 1 _ _
* 2 zapisywana 1 _ _
* 2 małymi 1 _ _
* 2 literami 1 _ _
* 2 oraz 1 _ _
* 2 odczytywana 1 _ _
* 2 jako 1 _ _
* 2 jedno 1 _ _
* 2 słowo 1 _ _
* 2 awk 1 _ _
* 2 . 0 _ _

[Alfred V. Aho] and [Brian Kernighan] have their corresponding Wikidata IDs, since it was possible to determine them using the Wikipedia and Wikidata datasets. Peter Weinberger does not have the ID, even though there is an entry in Wikidata about him. Yet, there is no such article in the Polish Wikipedia and the link could not be established automatically. In the test set only the items that have the corresponding Polish Wikipedia articles will have to be determined. Moreover, the algorithm will only have to determine the target of the link, not the span, so for the previous example, the test data will look as follows (the fourth column is superfluous but kept for compatiblity with the training data):

* 2 Nazwa 1 _ _
* 2 języka 1 _ _
* 2 pochodzi 1 _ _
* 2 od 1 _ _
* 2 pierwszych 1 _ _
* 2 liter 1 _ _
* 2 nazwisk 1 _ _
* 2 jego 1 _ _
* 2 autorów 1 _ _
* 2 Alfreda 1 _ e1
* 2 V 1 _ e1
* 2 . 0 _ e1
* 2 Aho 1 _ e1
* 2 , 0 _ _
* 2 Petera 1 _ _
* 2 Weinbergera 1 _ _
* 2 i 1 _ _
* 2 Briana 1 _ e2
* 2 Kernighana 1 _ e2
* 2 i 1 _ _
* 2 czasami 1 _ _
* 2 jest 1 _ _
* 2 zapisywana 1 _ _
* 2 małymi 1 _ _
* 2 literami 1 _ _
* 2 oraz 1 _ _
* 2 odczytywana 1 _ _
* 2 jako 1 _ _
* 2 jedno 1 _ _
* 2 słowo 1 _ _
* 2 awk 1 _ _
* 2 . 0 _ _

Thus the algorithm will be informed about 2 mentions, one spanning [Alfred V. Aho] and another spanning [Briana Kernighana]. It should be noted that in the test data mentions linking to the same entities will have separate mention IDs, unless they form a continuous span of tokens.

The second dataset is an extension of the first dataset with the additional columns:
[doc_id, token, lemma, preceding_space, morphosyntactic_tags, link_title, entity_id], e.g.

* 2 Nazwa nazwa 0 subst:sg:nom:f _ _
* 2 języka język 1 subst:sg:gen:m3 _ _
* 2 pochodzi pochodzić 1 fin:sg:ter:imperf _ _
* 2 od od 1 prep:gen:nwok _ _
* 2 pierwszych pierwszy 1 adj:pl:gen:f:pos _ _
* 2 liter litera 1 subst:pl:gen:f _ _
* 2 nazwisk nazwisko 1 subst:pl:gen:n _ _
* 2 jego on 1 ppron3:sg:gen:m1:ter:akc:npraep _ _
* 2 autorów autor 1 subst:pl:gen:m1 _ _
* 2 Alfreda Alfred 1 subst:sg:gen:m1 Alfred V. Aho Q62898
* 2 V V 1 subst:sg:nom:n Alfred V. Aho Q62898
* 2 . . 0 interp Alfred V. Aho Q62898
* 2 Aho Aho 1 subst:sg:gen:m1 Alfred V. Aho Q62898

## Test data
The test data is available here for DOWNLOAD.

## Evaluation
Since the mention spans will be provided in the test data, we will only use precision, i.e. the number of correctly identified mentions divided by the total number of mentions to be identified as the evaluation measure.

* Entity types
* human (https://www.wikidata.org/wiki/Q5)
* geographic location (https://www.wikidata.org/wiki/Q2221906)
* academic discipline (https://www.wikidata.org/wiki/Q11862829)
* anatomical structure (https://www.wikidata.org/wiki/Q4936952)
* occupation (https://www.wikidata.org/wiki/Q12737077)
* vehicle model (https://www.wikidata.org/wiki/Q29048322)
* construction (https://www.wikidata.org/wiki/Q811430)
* written work (https://www.wikidata.org/wiki/Q47461344)
* astronomical body (https://www.wikidata.org/wiki/Q6999)
* clothing (https://www.wikidata.org/wiki/Q11460)
* taxon (https://www.wikidata.org/wiki/Q16521)
* mythical entity (https://www.wikidata.org/wiki/Q24334685)
* type of sport (https://www.wikidata.org/wiki/Q31629)
* supernatural being (https://www.wikidata.org/wiki/Q28855038)
* liquid (https://www.wikidata.org/wiki/Q11435)
* political system (https://www.wikidata.org/wiki/Q28108)
* group of living things (https://www.wikidata.org/wiki/Q16334298)
* chemical entity (https://www.wikidata.org/wiki/Q43460564)
* publication (https://www.wikidata.org/wiki/Q732577)
* landform (https://www.wikidata.org/wiki/Q271669)
* language (https://www.wikidata.org/wiki/Q34770)
* unit (https://www.wikidata.org/wiki/Q2198779)
* physico-geographical object (https://www.wikidata.org/wiki/Q20719696)
* intellectual work (https://www.wikidata.org/wiki/Q15621286)
* tool (https://www.wikidata.org/wiki/Q39546)
* organism (https://www.wikidata.org/wiki/Q7239)
* food (https://www.wikidata.org/wiki/Q2095)
