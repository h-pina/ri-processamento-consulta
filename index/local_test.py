from index.structure import HashIndex
from index.indexer import HTMLIndexer

index = HashIndex()
indexador_teste = HTMLIndexer(index)
#o HTML está mal formado de propósito ;)
indexador_teste.index_text(10,"<strong>Ol&aacute;! </str> Quais são os dados que precisará?")

print(indexador_teste.index.dic_index)