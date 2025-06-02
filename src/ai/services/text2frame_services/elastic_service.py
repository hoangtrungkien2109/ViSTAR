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
        self.es.indices.put_settings(index="frame", settings={"index.blocks.read_only_allow_delete": None})
     
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

    def search(self, word: str, max_size: int = 5, user_id: str = "default") -> list[dict]:
        """Search similar words in Elasticsearch with priority to admin"""
        def _search_by_user(user_id):
            body = {
                "query": {
                    "bool": {
                        "must": [
                            {"match": {"word": word}},
                            {"match": {"user_id": user_id}}
                        ]
                    }
                },
                "size": max_size
            }
            return self.es.search(index="frame", body=body)["hits"]["hits"]

        # Search admin first
        admin_hits = _search_by_user("2")

        # If admin found results, check for default
        if admin_hits:
            default_hits = _search_by_user("default")
            # Compare _id of results
            admin_ids = {hit["_id"] for hit in admin_hits}
            default_ids = {hit["_id"] for hit in default_hits}
            if admin_ids == default_ids:
                return admin_hits  # identical -> return only admin
            return admin_hits + [hit for hit in default_hits if hit["_id"] not in admin_ids]
        else:
            return _search_by_user("default")


    def upload_to_es(self, mapping_path: str, data_path: str,
                     json_path: str | None = None, user_id: str = "default"):
        """Upload words and their frames into elasticsearch database"""
        words, file_names = self._process_data(mapping_path)
        frame_chunks = []
        for file_name in file_names:
            try:
                frame_chunks.append((np.load(data_path + f'/landmarks_{file_name}.npy') * 1000).astype(np.int16).tolist())
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
                        "user_id": user_id
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
    
    def upload_one_to_es(self, 
                    word: str,
                    file_name: str,
                    frame: np.ndarray,
                    user_id):
        """Upload words and their frames into elasticsearch database"""
        data ={
                "_index": "frame",
                "_id": str(uuid.uuid4()),
                "_source": {
                    "word": word,
                    "frame": self._encode_frame(frame.tolist()),
                    "file_name": file_name,
                    "user_id": user_id
                }
        }
        try:
            print(data)
            success, errors = helpers.bulk(self.es, [data], raise_on_error=False, stats_only=False)
            print(f"Success: {success}, Errors: {errors}")
        except Exception as e:
            print(f"Bulk insert failed: {e}")
        logger.info("Data pushed to elastic successfully")

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
    result = es.search(word="hien",user_id="2")
    
    for hit in result:
        print("Word:", hit["_source"]["word"], 
              "\nfilename:", hit["_source"]["file_name"],
              "\nuser_id:", hit["_source"]["user_id"])
        # print("Frame:", es.decode_frame(hit["_source"]["frame"]))
    