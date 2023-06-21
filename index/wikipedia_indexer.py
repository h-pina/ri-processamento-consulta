from index.indexer import *
from index.structure import *
import time


if __name__ == "__main__":
    index = HashIndex()
    html = HTMLIndexer(index)
    html.cleaner = Cleaner(stop_words_file="stopwords.txt",
                        language="portuguese",
                        perform_stop_words_removal=True,
                        perform_accents_removal=True,
                        perform_stemming=False)
    startTime = time.time()
    html.index_text_dir("index/wiki_data")
    index.write("wiki.idx")
    endTime = time.time()
    print(f"Time spent: \n - {(endTime-startTime)/60} minutes \n - ({endTime-startTime} seconds)")
    
    