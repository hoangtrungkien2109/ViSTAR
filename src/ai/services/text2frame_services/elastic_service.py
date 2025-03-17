"""Elasticsearch class"""

import uuid
import json
import subprocess
import pandas as pd
import numpy as np
from loguru import logger
from elasticsearch import helpers, Elasticsearch
COORD_JOINER = "C"
POINT_JOINER = "P"
FRAME_JOINER = "F"


class ESEngine():
    """Class for Elastic Search"""
    _instance = None

    def __new__(cls, *args, **kwargs) -> None:
        if cls._instance is None:
            cls._instance = super(ESEngine, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self, index_name: str = "frame"):
        self.index_name = index_name
        self.es = Elasticsearch("http://localhost:9200", request_timeout=60)
        if not self.es.indices.exists(index=self.index_name):
            self.es.indices.create(index=self.index_name)
            logger.warning("INDEX CREATED")

    def _encode_frame(self, frames: list) -> str:
        """Convert array frame to a single string"""
        frame_list = []
        for frame in frames:
            point_list = []
            for point in frame:
                coord_list = [str(coord) for coord in point]
                point_list.append(COORD_JOINER.join(coord_list))
            frame_list.append(POINT_JOINER.join(point_list))
        return FRAME_JOINER.join(frame_list)

    def decode_frame(self, str_frame: str) -> list:
        if not str_frame or not isinstance(str_frame, str):
            return []
            
        frame_list = []
        str_frame_list = str_frame.split(FRAME_JOINER)
        
        for frame in str_frame_list:
            if not frame:  # Skip empty frames
                continue
                
            point_list = []
            str_point_list = frame.split(POINT_JOINER)
            
            for point in str_point_list:
                if not point:  # Skip empty points
                    continue
                    
                str_coord_list = point.split(COORD_JOINER)
                coord_list = []
                
                for coord in str_coord_list:
                    try:
                        if coord.strip():  # Only convert non-empty strings
                            coord_list.append(float(coord.strip()))
                    except ValueError as e:
                        logger.warning(f"Invalid coordinate value: {coord}")
                        continue
                        
                if coord_list:  # Only add points that have valid coordinates
                    point_list.append(coord_list)
                    
            if point_list:  # Only add frames that have valid points
                frame_list.append(point_list)
                
        return frame_list

    def _process_data(self, file_path: str) -> None:
        """Process the raw data so that it can be pushed to elastic"""
        ds = pd.read_csv(file_path)
        filenames = ds.ID.values
        words = ds.Word.values
        processed_words = []
        processed_files = []
        for word, _file in zip(words, filenames):
            new_word = word.lower()
            new_word = new_word.split("(")[0]
            new_word = new_word.strip()
            if new_word in processed_words:
                continue
            else:
                processed_words.append(new_word)
                processed_files.append(_file)
        return processed_words, processed_files

    def search(self, word: str) -> list[dict]:
        """Search similar words in elasticsearch"""
        search_body = {
            "query": {
                "fuzzy": {
                    "word": {
                        "value": word,
                        "fuzziness": "AUTO"
                    }
                }
            }
        }
        result = self.es.search(index = "frame", body=search_body)
        if len(result["hits"]["hits"]) > 0:  # FIX: vì sao lại >0 mà không phải ==0
            search_body = {
                "query": {
                    "match": {
                        "word": word,
                    }
                }
            }
        result = self.es.search(index = "frame", body=search_body)
        return result["hits"]["hits"]

    def upload_to_es(self, mapping_path: str, data_path: str,
                     json_path: str | None = None):
        """Upload words and their frames into elasticsearch database"""
        words, file_names = self._process_data(mapping_path)
        frame_chunks = []
        for file_name in file_names:
            try:
                frame_chunks.append(np.load(data_path + f'/landmarks_{file_name}.npy').tolist())
            except FileNotFoundError:
                logger.error(file_name)
        data = []
        logger.info(f"Length of frame_chunks: {len(frame_chunks)}")
        for word, frame, file_name in zip(words, frame_chunks, file_names):
            data.append(
                {
                    "_index": self.index_name,
                    "_id": str(uuid.uuid4()),
                    "_source": {
                        "word": word,
                        "frame": self._encode_frame(frame),
                        "file_name": file_name,
                    }
                }
            )
            # data = {
            #     "word": word,
            #     "frame": frame,
            #     "file_name": file_name,
            # }
            # _id = str(uuid.uuid4())
            # try:
            #     self.es.index(index=self.index_name, id=_id, document=data)
            # except Exception as e:
            #     logger.debug(e)
            #     logger.warning(f"Word: {word}, File: {file_name}")
        logger.warning("Starting to upload to elasticsearch")
        for i in range(100):
            helpers.bulk(self.es, data[int(len(data)/100*i):int(min(len(data), (len(data)/100*(i+1))))])
        logger.info("Data pushed to elastic successfully")
        if json_path is not None:
            with open(json_path, "a") as f:
                for doc in data:
                    json.dump(doc, f)

    def clear_data_es(self):
        """Clear all data from elasticsearch"""
        curl_command = [
            'curl', 
            '-X', 
            'DELETE', 
            f'http://localhost:9200/{self.index_name}'
        ]
        subprocess.run(curl_command, check=True)
        logger.info("Successfully delete all data")

    def delete_index(self):
        """Delete the current index"""
        try:
            self.es.indices.delete(index=self.index_name)
            print(f"Deleted old index '{self.index_name}'.")
        except Exception as e:
            logger.warning(e)
            print(f"Index '{self.index_name}' not found. Skipping deletion.")


if __name__ == "__main__":
    es: ESEngine = ESEngine()
    print(es.search("kiến"))