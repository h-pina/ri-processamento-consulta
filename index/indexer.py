from nltk.stem.snowball import SnowballStemmer
from bs4 import BeautifulSoup
import string
from tqdm import tqdm
from nltk.tokenize import word_tokenize
import os
from multiprocessing import Process


class Cleaner:
    def __init__(self, stop_words_file: str, language: str,
                 perform_stop_words_removal: bool, perform_accents_removal: bool,
                 perform_stemming: bool):
        self.set_stop_words = self.read_stop_words(stop_words_file)

        self.stemmer = SnowballStemmer(language)
        in_table = "áéíóúâêôçãẽõü"
        out_table = "aeiouaeocaeou"
        
        self.accents_translation_table = str.maketrans(in_table, out_table)
        self.set_punctuation = set(string.punctuation)

        # flags
        self.perform_stop_words_removal = perform_stop_words_removal
        self.perform_accents_removal = perform_accents_removal
        self.perform_stemming = perform_stemming

    def html_to_plain_text(self, html_doc: str) -> str:
        soup = BeautifulSoup(html_doc, 'html.parser')
        return soup.get_text()

    @staticmethod
    def read_stop_words(str_file) -> set:
        set_stop_words = set()
        with open(str_file, encoding='utf-8') as stop_words_file:
            for line in stop_words_file:
                arr_words = line.split(",")
                [set_stop_words.add(word) for word in arr_words]
        return set_stop_words

    def is_stop_word(self, term: str):
        return term in self.set_stop_words 

    def word_stem(self, term: str):
        return self.stemmer.stem(term)

    def remove_accents(self, term: str) -> str:
        return term.translate(self.accents_translation_table)

    def preprocess_word(self, term: str) -> str or None:
        if term in self.set_punctuation:
            return None
        if self.perform_stop_words_removal and self.is_stop_word(term):
            return None
        return self.word_stem(term)
    
    def preprocess_text(self, text: str) -> str or None:
        text = text.lower()
        return text.translate(self.accents_translation_table)

class HTMLIndexer:
    cleaner = Cleaner(stop_words_file="stopwords.txt",
                        language="portuguese",
                        perform_stop_words_removal=True,
                        perform_accents_removal=True,
                        perform_stemming=True)

    def __init__(self, index):
        self.index = index

    def text_word_count(self, plain_text: str):
        dic_word_count = {}
        cleanText = self.cleaner.preprocess_text(plain_text)
        tokens = word_tokenize(cleanText)
        for token in tokens:
            checkedToken = self.cleaner.preprocess_word(token)
            if checkedToken:
                dic_word_count[checkedToken] = tokens.count(token)
        return dic_word_count

    def index_text(self, doc_id: int, text_html: str):
        cleanText = self.cleaner.html_to_plain_text(text_html)
        dict_count = self.text_word_count(cleanText)
        for term in dict_count:
            self.index.index(term,doc_id,dict_count[term])
        self.index.finish_indexing()


    def index_text_dir(self, path: str):
        for str_sub_dir in tqdm(os.listdir(path)):
            path_sub_dir = f"{path}/{str_sub_dir}"
            for file in os.listdir(path_sub_dir):
                with open(f"{path_sub_dir}/{file}",'r',encoding='utf-8') as f:
                    pureHtml = f.read()
                    doc_id = file.replace(".html","")
                    self.index_text(doc_id,pureHtml)