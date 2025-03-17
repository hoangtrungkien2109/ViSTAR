from pydantic import BaseModel
from pyvi import ViTokenizer
from loguru import logger
from uuid import uuid4
from typing import List


class Word(BaseModel):
    word: str = ""
    is_name: bool = False
    segment_id: int = None

class Segment(BaseModel):
    segment: str = ""
    is_name: bool = False
    segment_id: int = None

class WordList(BaseModel):
    segment_list: List[Segment] = []
    special_word_list: List[Word] = []
    pointer: int = 0

    def get_raw_len(self):
        return len(self.special_word_list)

    def get_segment_len(self):
        return len(self.segment_list)

    def get_sentence(self):
        return " ".join([segment.segment for segment in self.segment_list])

    def put(self, word: Word, index: int = -1):
        if index == -1:
            self.special_word_list.append(Word(word=word))
            self.segment_list.append(Segment(segment=word))
        else:
            self.special_word_list.insert(index, Word(word=word))
            self.segment_list.insert(index, Segment(segment=word))

    def get(self, index: int = 0):
        return self.segment_list[index]

    def get_raw(self, index: int = 0):
        return self.special_word_list[index]

    def pop(self, index: int = 0):
        current_segment = self.segment_list[index]
        for idx in range(len(self.special_word_list)):
            flag = False
            while self.special_word_list[idx].segment_id == current_segment.segment_id:
                self.special_word_list.pop(idx)
                flag = True
            if flag:
                break
        return self.segment_list.pop(index)

    def clear_old_segments(self):
        self.segment_list = []

    def tokenize_text(self):
        """Tokenize text"""
        segments = ViTokenizer.tokenize(self.get_sentence()).split(" ")
        self.clear_old_segments()
        total_word = 0
        for idx, segment in enumerate(segments):
            sub_segment = segment.split("_")
            # print(sub_segment)
            self.segment_list.append(Segment(
                segment=" ".join(sub_segment),
                segment_id=idx,
                is_name=self.special_word_list[total_word].is_name
            ))

            while len(sub_segment)>0:
                self.special_word_list[total_word].segment_id = idx
                sub_segment.pop(0)
                total_word += 1
