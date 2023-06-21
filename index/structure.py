from IPython.display import clear_output
from typing import List, Set, Union
from abc import abstractmethod
from functools import total_ordering
from os import path
import os
import pickle
import gc


class Index:
    def __init__(self):
        self.dic_index = {}
        self.set_documents = set()

    def index(self, term: str, doc_id: int, term_freq: int):
        if term not in self.dic_index:
            int_term_id = len(self.dic_index)
            self.dic_index[term] = self.create_index_entry(int_term_id)
        else:
            int_term_id = self.get_term_id(term)

        self.add_index_occur(self.dic_index[term], doc_id, int_term_id, term_freq)
        self.set_documents.add(doc_id) 

    @property
    def vocabulary(self) -> List[str]:
        return list(self.dic_index.keys())

    @property
    def document_count(self) -> int:
        return len(self.set_documents)

    @abstractmethod
    def get_term_id(self, term: str):
        raise NotImplementedError("Voce deve criar uma subclasse e a mesma deve sobrepor este método")

    @abstractmethod
    def create_index_entry(self, termo_id: int):
        raise NotImplementedError("Voce deve criar uma subclasse e a mesma deve sobrepor este método")

    @abstractmethod
    def add_index_occur(self, entry_dic_index, doc_id: int, term_id: int, freq_termo: int):
        raise NotImplementedError("Voce deve criar uma subclasse e a mesma deve sobrepor este método")

    @abstractmethod
    def get_occurrence_list(self, term: str) -> List:
        raise NotImplementedError("Voce deve criar uma subclasse e a mesma deve sobrepor este método")

    @abstractmethod
    def document_count_with_term(self, term: str) -> int:
        raise NotImplementedError("Voce deve criar uma subclasse e a mesma deve sobrepor este método")

    def finish_indexing(self):
        pass

    def write(self, arq_index: str):
        with open(arq_index, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def read(arq_index: str):
        idx = {}
        with open(arq_index, 'rb') as f:
            idx = pickle.load(f)
        return idx

    def __str__(self):
        arr_index = []
        for str_term in self.vocabulary:
            arr_index.append(f"{str_term} -> {self.get_occurrence_list(str_term)}")

        return "\n".join(arr_index)

    def __repr__(self):
        return str(self)


@total_ordering
class TermOccurrence:
    def __init__(self, doc_id: int, term_id: int, term_freq: int):
        self.doc_id =  int(doc_id) if doc_id is not None else 0
        self.term_id = int(term_id)
        self.term_freq = int(term_freq)

    def write(self, idx_file):
        idx_file.write(self.doc_id.to_bytes(4,'big'))
        idx_file.write(self.term_id.to_bytes(4,'big'))
        idx_file.write(self.term_freq.to_bytes(4,'big'))
        

    def __hash__(self):
        return hash((self.doc_id, self.term_id))

    def __eq__(self, other_occurrence: "TermOccurrence"):
        if other_occurrence is not None:
            return (self.term_id == other_occurrence.term_id) and (self.doc_id == other_occurrence.doc_id)
        else:
            return False

    def __lt__(self, other_occurrence: "TermOccurrence"): 
        if other_occurrence is not None:
            if self.doc_id != other_occurrence.doc_id:
                return self.doc_id < other_occurrence.doc_id 
            else:
                return self.term_id < other_occurrence.term_id 
                    
        else:
            return True

    def __str__(self):
        return f"( doc: {self.doc_id} term_id:{self.term_id} freq: {self.term_freq})"

    def __repr__(self):
        return str(self)


# HashIndex é subclasse de Index
class HashIndex(Index):
    def get_term_id(self, term: str):
        return self.dic_index[term][0].term_id

    def create_index_entry(self, termo_id: int) -> List:
        return []

    def add_index_occur(self, entry_dic_index: List[TermOccurrence], doc_id: int, term_id: int, term_freq: int):
        entry_dic_index.append(TermOccurrence(doc_id,term_id,term_freq))

    def get_occurrence_list(self, term: str) -> List:
        return self.dic_index[term] if term in self.dic_index else []

    def document_count_with_term(self, term: str) -> int:
        return len(self.dic_index[term]) if term in self.dic_index else 0


class TermFilePosition:
    def __init__(self, term_id: int, term_file_start_pos: int = None, doc_count_with_term: int = None):
        self.term_id = term_id

        # a serem definidos após a indexação
        self.term_file_start_pos = term_file_start_pos
        self.doc_count_with_term = doc_count_with_term

    def __str__(self):
        return f"term_id: {self.term_id}, doc_count_with_term: {self.doc_count_with_term}, term_file_start_pos: {self.term_file_start_pos}"

    def __repr__(self):
        return str(self)


class FileIndex(Index):
    TMP_OCCURRENCES_LIMIT = 1000000

    def __init__(self):
        super().__init__()

        self.lst_occurrences_tmp = [None]*FileIndex.TMP_OCCURRENCES_LIMIT
        self.idx_file_counter = 0
        self.str_idx_file_name = None

        # metodos auxiliares para verifica o tamanho da lst_occurrences_tmp
        self.idx_tmp_occur_last_element  = -1
        self.idx_tmp_occur_first_element = 0
        
    def get_tmp_occur_size(self):
        return self.idx_tmp_occur_last_element - self.idx_tmp_occur_first_element + 1

    def get_term_id(self, term: str):
        return self.dic_index[term].term_id

    def create_index_entry(self, term_id: int) -> TermFilePosition:
        return TermFilePosition(term_id)

    def add_index_occur(self, entry_dic_index: TermFilePosition, doc_id: int, term_id: int, term_freq: int):
        self.idx_tmp_occur_last_element += 1
        self.lst_occurrences_tmp[self.idx_tmp_occur_last_element] = TermOccurrence(doc_id,term_id,term_freq) 
        if self.idx_tmp_occur_last_element == self.TMP_OCCURRENCES_LIMIT:

            self.save_tmp_occurrences()

    def next_from_list(self) -> TermOccurrence:
        if self.get_tmp_occur_size() > 0:
            next_occur = self.lst_occurrences_tmp[self.idx_tmp_occur_first_element]
            self.idx_tmp_occur_first_element+=1
            return next_occur

        else:
            return None

    def next_from_file(self, file_pointer) -> TermOccurrence:
        # next_from_file = pickle.load(file_idx) TODO: O que e isso
        bytes_doc_id = file_pointer.read(4)
        if not bytes_doc_id:
            return None
        bytes_term_id = file_pointer.read(4)
        if not bytes_term_id:
            return None
        bytes_term_freq = file_pointer.read(4)
        if not bytes_term_freq :
            return None
        
        doc_id = int.from_bytes(bytes_doc_id, "big")
        term_id = int.from_bytes(bytes_term_id, "big")
        term_freq = int.from_bytes(bytes_term_freq, "big")
        
        return TermOccurrence(doc_id, term_id, term_freq)

    def save_tmp_occurrences(self):
        # Ordena pelo term_id, doc_id
        #    Para eficiência, todo o código deve ser feito com o garbage collector desabilitado gc.disable()
        gc.disable()
        self.lst_occurrences_tmp.sort(key=lambda x:(x is None, x))
    
        #If there is no index file created yet, create one and fill with the list
        if self.str_idx_file_name == None: 
            self.str_idx_file_name = "occur_index_" + str(self.idx_file_counter)
            with open(self.str_idx_file_name, 'wb') as file:
                next = self.next_from_list()
                while(next is not None):
                    next.write(file)
                    next = self.next_from_list()

        else:
            old_file_name = self.str_idx_file_name
            self.idx_file_counter += 1
            self.str_idx_file_name = "occur_index_" + str(self.idx_file_counter)

            with open( self.str_idx_file_name ,'wb') as new_file:
                with open(old_file_name,'rb') as old_file:
                    next_file = self.next_from_file(old_file)
                    next_list =  self.next_from_list()
                    while(next_file is not None):
                        if(next_file < next_list):
                            next_file.write(new_file)
                            next_file = self.next_from_file(old_file)
                        else:
                            next_list.write(new_file)
                            next_list =  self.next_from_list()
                    while(next_list is not None):
                        next_list.write(new_file)
                        next_list =  self.next_from_list()
            os.remove(old_file_name)
        gc.enable()

        self.idx_tmp_occur_last_element  = -1
        self.idx_tmp_occur_first_element = 0

    def finish_indexing(self):
        if len(self.lst_occurrences_tmp) > 0:
            self.save_tmp_occurrences()
        # Sugestão: faça a navegação e obetenha um mapeamento
        # id_termo -> obj_termo armazene-o em dic_ids_por_termo
        # obj_termo é a instancia TermFilePosition correspondente ao id_termo
        dic_ids_por_termo = {}
        for str_term, obj_term in self.dic_index.items():
            dic_ids_por_termo[obj_term.term_id] = obj_term

        with open(self.str_idx_file_name, 'rb') as idx_file:
            # navega nas ocorrencias para atualizar cada termo em dic_ids_por_termo
            # apropriadamente
            next_occur = self.next_from_file(idx_file)
            size = idx_file.tell()
            pos = 0
            while(next_occur is not None):
                obj_term = dic_ids_por_termo[next_occur.term_id]
                if(obj_term.term_file_start_pos is None):
                    obj_term.term_file_start_pos = pos*size
                if(obj_term.doc_count_with_term  is None):
                    obj_term.doc_count_with_term = 1
                else:
                    obj_term.doc_count_with_term += 1
                
                dic_ids_por_termo[next_occur.term_id] = obj_term
                next_occur = self.next_from_file(idx_file)
                pos+=1

    def get_occurrence_list(self, term: str) -> List:
        if term in self.dic_index:
            occurences = []
            with open(self.str_idx_file_name,'rb') as file:
                next_occur = self.next_from_file(file)
                while(next_occur is not None):
                    if next_occur.term_id == self.dic_index[term].term_id:
                        occurences.append(next_occur)
                    next_occur = self.next_from_file(file)
            print()
            return occurences
        else:
            return []

    def document_count_with_term(self, term: str) -> int:
        return self.dic_index[term].doc_count_with_term  if term in self.dic_index else 0
