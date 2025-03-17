"""
File contains function and class related to similarity sentence
Aim to convert from a paragraph of raw text to a list of existing words from database
"""

import re
# import time
import json
from typing import List, Dict
# import orjson
import torch
import numpy as np
from loguru import logger
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
# from sentence_transformers import SentenceTransformer
# from sentence_transformers import SentenceTransformer
from src.ai.services.text2frame_services.elastic_service import ESEngine
from src.ai.services.utils.decorator import processing_time
from src.ai.services.text2frame_services.models.custom_list import WordList, Word, Segment
import random

class SimilaritySentence():
    """Class for Elastic Search"""
    # _instance = None

    # def __new__(cls, *args, **kwargs) -> None:
    #     if cls._instance is None:
    #         cls._instance = super(SimilaritySentence, cls).__new__(cls, *args, **kwargs)
    #     return cls._instance

    def __init__(self,
            default_dict_path: str = "D:/NCKH/Text_to_Sign/ViSTAR/src/ai/services/text2frame_services/data/character_dict.rar",
            ner_model_name: str = "NlpHUST/ner-vietnamese-electra-base",
            ner_min_length: int = 5,
            ner_max_length: int = 20,
        ):
        self.es: ESEngine = ESEngine()
        self.word_queue: List[Segment] = []
        self.ner_list: WordList = WordList()
        self.ner_min_length = ner_min_length
        self.ner_max_length = ner_max_length
        self.has_default = False

        tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
        ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_name)
        self.ner = pipeline("ner", model=ner_model, tokenizer=tokenizer,
                       device='cuda' if torch.cuda.is_available() else 'cpu')
        try:
            with open(default_dict_path, 'r') as f:
                self.default_frame = json.load(f)
                logger.info(f"Default character: {self.default_frame.keys()}")
        except FileNotFoundError as e:
            raise e

    def clean_text(self, text: str) -> str:
        """Clean raw text"""
        text = re.sub(r"[-()\"#/@;:<>{}=~|.?,]", "", text)
        return text

    def push_word(self, word: str) -> None:
        old_length = self.ner_list.get_segment_len()
        self.ner_list.put(word=word)
        self.ner_list.tokenize_text()
        new_length = self.ner_list.get_segment_len()
        logger.debug(f"Len of word list: {self.ner_list.get_segment_len()}")
        # logger.warning(f"Word raw list: {self.ner_list.special_word_list}")
        # logger.warning(f"Word segment list: {self.ner_list.segment_list}")
        logger.debug(f"Word list: {self.ner_list.get_sentence()}")
        if self.ner_list.get_segment_len() > self.ner_min_length:
            self._detect_name()
            if self.ner_list.get_segment_len() > self.ner_max_length:
                self.ner_list.pop()
                self.ner_list.pointer -= 1
            if old_length != new_length:
                self.word_queue.append(self.ner_list.get(self.ner_list.pointer))
                self.ner_list.pointer += 1

    @processing_time
    def get_frame(self) -> List[np.ndarray]:
        if len(self.word_queue) > 0:
            current_word = self.word_queue.pop()
            if current_word.is_name == True:
                logger.success("Sent a name (list of character) to streaming")
                frames = []
                for character in self.remove_accents(current_word.segment.replace(" ","")):
                    frames.extend(self.default_frame[character])
                logger.error(np.shape(frames))
                self.has_default = False
                return frames
            else:
                searched_result = self.es.search(word=current_word.segment)
                logger.debug(f"CURRENT: {current_word.segment}")
                if len(searched_result) > 0:
                    logger.success("Sent frame to streaming")
                    frames = self.es.decode_frame(searched_result[0]["_source"]["frame"])
                    self.has_default = False
                    return frames
                # elif not self.has_default:
                else:
                    logger.error("Word is not contained in DB")
                    self.has_default = True
                    word = random.choice(["a", "b", "c", "d", "e", "g", "h"])
                    logger.info(f"Searched: {word}")
                    return self.default_frame[word]
        # elif not self.has_default:
        #     logger.error("Default")
        #     self.has_default = True
        #     return self.default_frame["default"]
        else:
            logger.error("Word is not contained in DB")
            self.has_default = True
            word = random.choice(["a", "b", "c", "d", "e", "g", "h"])
            logger.info(f"Searched: {word}")
            return self.default_frame[word]

    @processing_time
    def _detect_name(self) -> None:
        """Detect all entity in sentence"""
        entities = self.ner(self.ner_list.get_sentence())
        for entity in entities:
            if "PERSON" in entity['entity']:
                word_list = [word.word for word in self.ner_list.special_word_list]
                for i in range(len(word_list)):
                    if entity["index"]-1 == i:
                        self.ner_list.get_raw(i).is_name = True
                        logger.success(f"Name: {self.ner_list.get_raw(i).word}")
                        break
            # for idx, word in enumerate([word.word for word in self.ner_list.word_list]):
                
            # if "PERSON" in entity['entity']:
            #     self.ner_list.get(entity['index'] - 1).is_name = True
            #     logger.success(f"Name: {self.ner_list.get(entity['index'] - 1).word}")

    def remove_accents(self, old: str):
        """
        Removes common accent characters, lower form.
        Uses: regex.
        """
        new = old.lower()
        new = re.sub(r'[àáảãạ]', 'a', new)
        new = re.sub(r'[ằắẳẵặă]', 'ă', new)
        new = re.sub(r'[ầấẩẫậâ]', 'â', new)
        new = re.sub(r'[èéẻẽẹ]', 'e', new)
        new = re.sub(r'[ềếểễệê]', 'ê', new)
        new = re.sub(r'[ìíỉĩị]', 'i', new)
        new = re.sub(r'[òóỏõọ]', 'o', new)
        new = re.sub(r'[ồốổỗộô]', 'o', new)
        new = re.sub(r'[ồốổỗộô]', 'o', new)
        new = re.sub(r'[ùúủũụ]', 'u', new)
        new = re.sub(r'[ừứửữựư]', 'ư', new)
        new = re.sub(r'[ỳýỷỹỵy]', 'y', new)
        return new

# ss = SimilaritySentence()
