from typing import List
from abc import abstractmethod
from typing import List, Set, Mapping
from index.structure import TermOccurrence
import math
from enum import Enum


class IndexPreComputedVals:
    def __init__(self, index):
        self.index = index
        self.precompute_vals()

    def precompute_vals(self):
        """
        Inicializa os atributos por meio do indice (idx):
            doc_count: o numero de documentos que o indice possui
            document_norm: A norma por documento (cada termo é presentado pelo seu peso (tfxidf))
        """
        self.document_norm = {}
        tf_id_lists_per_doc = {}
        self.doc_count = self.index.document_count
        for doc_id in range(0, self.doc_count):
            tf_id_lists_per_doc[doc_id + 1] = []

        for word in self.index.vocabulary:
            occurence_list = self.index.get_occurrence_list(word)
            for doc_id in range(1, self.doc_count + 1):
                word_occurence_in_doc = list(
                    filter(lambda x: x.doc_id == doc_id, occurence_list)
                )
                if len(word_occurence_in_doc) > 0:
                    doc_count = self.doc_count
                    term_freq = word_occurence_in_doc[0].term_freq
                    num_docs_with_term = self.index.document_count_with_term(word)
                    tf_id_lists_per_doc[doc_id].append(
                        VectorRankingModel.tf_idf(
                            doc_count, term_freq, num_docs_with_term
                        )
                    )

        for doc_id in tf_id_lists_per_doc.keys():
            tf_idf_array = tf_id_lists_per_doc[doc_id]
            total = sum([math.pow(x, 2) for x in tf_idf_array])
            result_for_doc = math.sqrt(total)
            self.document_norm[doc_id] = result_for_doc


class RankingModel:
    @abstractmethod
    def get_ordered_docs(
        self,
        query: Mapping[str, TermOccurrence],
        docs_occur_per_term: Mapping[str, List[TermOccurrence]],
    ):
        raise NotImplementedError(
            "Voce deve criar uma subclasse e a mesma deve sobrepor este método"
        )

    def rank_document_ids(self, documents_weight):
        doc_ids = list(documents_weight.keys())
        doc_ids.sort(key=lambda x: -documents_weight[x])
        return doc_ids


class OPERATOR(Enum):
    AND = 1
    OR = 2


# Atividade 1
class BooleanRankingModel(RankingModel):
    def __init__(self, operator: OPERATOR):
        self.operator = operator

    def get_set_list(
        self, map_lst_occurrences: Mapping[str, List[TermOccurrence]]
    ) -> List[int]:
        set_doc_ids = []
        for word_occurencies_list in map_lst_occurrences:
            doc_id_list = [
                term_occur.doc_id
                for term_occur in map_lst_occurrences[word_occurencies_list]
            ]
            set_doc_ids.append(set(doc_id_list))
        return set_doc_ids

    def intersection_all(
        self, map_lst_occurrences: Mapping[str, List[TermOccurrence]]
    ) -> List[int]:
        set_doc_ids = self.get_set_list(map_lst_occurrences)
        return set_doc_ids[0].intersection(*set_doc_ids) if set_doc_ids else {}

    def union_all(
        self, map_lst_occurrences: Mapping[str, List[TermOccurrence]]
    ) -> List[int]:
        set_doc_ids = self.get_set_list(map_lst_occurrences)
        return set_doc_ids[0].union(*set_doc_ids) if set_doc_ids else {}

    def get_ordered_docs(
        self,
        query: Mapping[str, TermOccurrence],
        map_lst_occurrences: Mapping[str, List[TermOccurrence]],
    ):
        """Considere que map_lst_occurrences possui as ocorrencias apenas dos termos que existem na consulta"""
        if self.operator == OPERATOR.AND:
            return self.intersection_all(map_lst_occurrences), None
        else:
            return self.union_all(map_lst_occurrences), None


# Atividade 2
class VectorRankingModel(RankingModel):
    def __init__(self, idx_pre_comp_vals: IndexPreComputedVals):
        self.idx_pre_comp_vals = idx_pre_comp_vals

    @staticmethod
    def tf(freq_term: int) -> float:
        return 1 + math.log2(freq_term)

    @staticmethod
    def idf(doc_count: int, num_docs_with_term: int) -> float:
        return math.log2(doc_count / num_docs_with_term)

    @staticmethod
    def tf_idf(doc_count: int, freq_term: int, num_docs_with_term) -> float:
        tf = VectorRankingModel.tf(freq_term)
        idf = VectorRankingModel.idf(doc_count, num_docs_with_term)
        return tf * idf

    def get_ordered_docs(
        self,
        query: Mapping[str, TermOccurrence],
        docs_occur_per_term: Mapping[str, List[TermOccurrence]],
    ):
        documents_weight = {}
        query_words = list(query.keys())
        documents_ids = set()
        for docList in docs_occur_per_term.values():
            documents_ids.update([doc.doc_id for doc in docList])

        for doc_id in documents_ids:
            accumulator = 0
            for query_word in query_words:
                try:
                    term_freq_on_doc = next(
                        x.term_freq
                        for x in docs_occur_per_term[query_word]
                        if x.doc_id == doc_id
                    )
                except:
                    term_freq_on_doc = 0.5

                tfidf_query = VectorRankingModel.tf_idf(
                    max(documents_ids),
                    query[query_word].term_freq,
                    len(docs_occur_per_term[query_word]),
                )
                tfidf_doc = VectorRankingModel.tf_idf(
                    max(documents_ids),
                    term_freq_on_doc,
                    len(docs_occur_per_term[query_word]),
                )

                accumulator += tfidf_doc * tfidf_query
            accumulator /= self.idx_pre_comp_vals.document_norm[doc_id]
            documents_weight[doc_id] = accumulator
        # retona a lista de doc ids ordenados de acordo com o TF IDF
        return self.rank_document_ids(documents_weight), documents_weight
