"""
File contains function and class related to similarity sentence
Aim to convert from a paragraph of raw text to a list of existing words from database
"""

import re
import time
import json
from threading import Thread
from typing import List, Dict
# import orjson
import torch
import numpy as np
from loguru import logger
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from src.ai.services.text2frame_services.elastic_service import ESEngine
from src.ai.services.utils.decorator import processing_time
from src.ai.services.text2frame_services.models.custom_list import WordList, Word, Segment


class SimilaritySentence():
    """Class for Elastic Search"""
    # _instance = None

    # def __new__(cls, *args, **kwargs) -> None:
    #     if cls._instance is None:
    #         cls._instance = super(SimilaritySentence, cls).__new__(cls, *args, **kwargs)
    #     return cls._instance

    def __init__(self,
            default_dict_path: str = "D:/NCKH/Text_to_Sign/ViSTAR/src/ai/services/text2frame_services/data/character_dict.json",
            ner_model_name: str = "NlpHUST/ner-vietnamese-electra-base",
            embed_model_name: str = "dangvantuan/vietnamese-embedding",
            ner_min_length: int = 5,
            ner_max_length: int = 20,
        ):
        self.es: ESEngine = ESEngine()
        self.word_queue: List[Segment] = []
        self.ner_list: WordList = WordList()
        self.ner_min_length = ner_min_length
        self.ner_max_length = ner_max_length
        self.max_time_length = 2.0
        self.has_default = False

        tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
        ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_name)
        self.embed_model = SentenceTransformer(embed_model_name)
        self.ner = pipeline("ner", model=ner_model, tokenizer=tokenizer,
                       device='cuda' if torch.cuda.is_available() else 'cpu')
        try:
            with open(default_dict_path, 'r') as f:
                self.default_frame = json.load(f)
                logger.info(f"Default character: {self.default_frame.keys()}")
        except FileNotFoundError as e:
            raise e
        self.current_time = time.time()
        temp_thread = Thread(target=self.thread_counting_to_push, daemon=True)
        temp_thread.start()

    def clean_text(self, text: str) -> str:
        """Clean raw text"""
        text = re.sub(r"[-()\"#/@;:<>{}=~|.?,]", "", text)
        return text

    def push_word(self, word: str) -> None:
        self.current_time = time.time()
        old_length = self.ner_list.get_segment_len()
        self.ner_list.put(word=word)
        self.ner_list.tokenize_text()
        new_length = self.ner_list.get_segment_len()
        logger.debug(f"Len of word list: {self.ner_list.get_segment_len()}")
        logger.debug(f"Word list: {self.ner_list.get_sentence()}")
        if self.ner_list.get_segment_len() - self.ner_list.pointer > self.ner_min_length:
            self._detect_name()
            if self.ner_list.get_segment_len() > self.ner_max_length:
                self.ner_list.pop()
                self.ner_list.pointer -= 1
            if old_length != new_length:
                self.word_queue.append(self.ner_list.get(self.ner_list.pointer))
                self.ner_list.pointer += 1

    @processing_time
    def get_frame(self, similarity_threshold = 0.5) -> List[np.ndarray]:
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
                # Search for the corresponding word
                searched_result = self.es.search(word=current_word.segment)
                logger.debug(f"CURRENT: {current_word.segment}")
                if len(searched_result) > 0:
                    logger.success("Sending frame to streaming")

                    # Embed the raw sentence
                    raw_sentence = self.ner_list.get_sentence()
                    raw_embedding = self.embed_model.encode([raw_sentence])

                    # Embed the new sentences that replace the old word with the corresponding words
                    updated_sentences: List = []
                    for result in searched_result:
                        updated_sentences.append(re.sub(current_word.segment, result["_source"]["word"], raw_sentence, count=1)) # FIX
                    updated_embeddings = self.embed_model.encode(updated_sentences)
                    similarities = self.embed_model.similarity(raw_embedding, updated_embeddings)[0]
                    similarities = similarities[similarities >= similarity_threshold]
                    if len(similarities) > 0:
                        max_index = np.argmax(similarities)
                        frames = self.es.decode_frame(searched_result[max_index]["_source"]["frame"])
                        self.has_default = False
                        return frames
                    else:
                        logger.warning("Corresponding word is not appropriate")
                        self.has_default = True
                        return self.default_frame["default"]
                elif not self.has_default:
                    logger.error("Word is not contained in DB")
                    self.has_default = True
                    return self.default_frame["default"]
        elif not self.has_default:
            logger.error("Default")
            self.has_default = True
            return self.default_frame["default"]

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

    def thread_counting_to_push(self):
        while(True):
                check_time = time.time()
                logger.info(check_time - self.current_time)
                if self.ner_list.get_segment_len() > self.ner_list.pointer:
                    logger.info(self.ner_list.get(self.ner_list.pointer))
                    if self.ner_list.get_segment_len() > self.ner_max_length:
                        self.ner_list.pop()
                        self.ner_list.pointer -= 1
                    if check_time - self.current_time > self.max_time_length:
                        self.word_queue.append(self.ner_list.get(self.ner_list.pointer))
                        logger.debug(f"Word list: {self.ner_list.get_sentence()}")
                        self.ner_list.pointer += 1
                        self.current_time = time.time()
                time.sleep(self.max_time_length / 4.0)

# ss = SimilaritySentence()
