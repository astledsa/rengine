import re
import nltk
from typing import Dict
import importlib.resources
from tantivy import Query, Occur, Index
from utils.models import ProcessedQuery
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
from symspellpy import SymSpell, Verbosity

# nltk_packages = [
#     ("corpora/wordnet", "wordnet"),
#     ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
#     ("tokenizers/punkt_tab", "punkt_tab"),
#     ("corpora/stopwords", "stopwords"),
# ]

# for path, name in nltk_packages:
#     try:
#         nltk.data.find(path)
#     except LookupError:
#         nltk.download(name)

_WORD = re.compile(r"\w+", re.UNICODE)

def _q(ix: Index, qstr: str) -> Query:
    # Let Tantivy analyze the string with the field's analyzer
    return ix.parse_query(qstr, ["content"])

def _sanitize_tokens(xs):
    out = []
    for x in xs or []:
        out.extend(_WORD.findall(str(x).lower()))
    return [t for t in out if t]

lemmatizer = WordNetLemmatizer()

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
with importlib.resources.path("symspellpy", "frequency_dictionary_en_82_765.txt") as dictionary_path:
    sym_spell.load_dictionary(str(dictionary_path), term_index=0, count_index=1)

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def structure_query (natural_language_query: str) -> ProcessedQuery:
    clean_query = natural_language_query.lower()
    raw_tokens = clean_query.split()

    corrected_tokens = []
    for token in raw_tokens:
        suggestions = sym_spell.lookup(token, Verbosity.CLOSEST, max_edit_distance=1)
        corrected_tokens.append(suggestions[0].term if suggestions else token)

    corrected_query = " ".join(corrected_tokens)
    tokens = nltk.word_tokenize(corrected_query)
    filtered_tokens = [t for t in tokens if t.isalpha() and t not in stopwords.words("english")]

    lemmas = [lemmatizer.lemmatize(tok, get_wordnet_pos(tok)) for tok in filtered_tokens]

    expanded_terms = set(lemmas)
    for lemma in lemmas:
        for syn in wordnet.synsets(lemma):
            for l in syn.lemmas():
                term = l.name().lower().replace("_", " ")
                if " " not in term and term != lemma:
                    expanded_terms.add(term)

    return ProcessedQuery(
        raw=raw_tokens,
        autocorrected=corrected_query,
        expanded_terms=sorted(expanded_terms - set(lemmas)),
        primary_keywords=sorted(set(lemmas))
    )

def build_query(ix: Index, q: ProcessedQuery) -> Query:

    qstr = " OR ".join(q.primary_keywords)
    return ix.parse_query(query=qstr, default_field_names=["content"])
    
