# -*- coding: utf-8 -*-
import time

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
from typing import List, Dict
import tenacity
from langchain.schema.language_model import BaseLanguageModel
from langchain_community.vectorstores.faiss import FAISS
from typing import Any, Optional, List

import logging
import json
import warnings


warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class ItemVector:
    def __init__(self, db_path: str, model_name: str, llm: BaseLanguageModel, data_path: str = None,
                 verbose: bool = True, force_create: bool = True, use_multi_query: bool = False) -> None:
        self.db_path = db_path
        self.embeddings = self.load_embeddings(model_name)
        if verbose:
            print(f"Load Embedding Model: {model_name} successful!")

        self.verbose = verbose
        
        # Check if index.faiss file exists (more specific check)
        index_path = os.path.join(db_path, 'index.faiss')
        
        if os.path.exists(index_path) and force_create is False:
            # 加载本地数据库
            if verbose:
                print(f"Dectect the DB: {db_path} is exists, Load the Local DB!")
            self.load_local_db()
        else:
            if verbose:
                print(f"Creating new database at: {db_path}")
            metadatas, text = self.load_data(data_path)
            self.create_db(metadatas=metadatas, text=text)

        self.retriever = self.db.as_retriever()

    def load_data(self, data_path):
        file_path = data_path
        with open(file_path, 'r', encoding='utf-8') as file:
            user_data = json.load(file)
        metadatas = []
        text = []
        for user_id, data in user_data.items():
            # Handle missing fields gracefully
            title = str(data.get('title', ''))
            description = str(data.get('description', ''))
            new_description = str(data.get('new_description', ''))
            categories = data.get('categories', [])
            features = data.get('features', [])
            
            # Create search text
            search_text = title + ' ' + new_description + ' ' + description + ' ' + str(categories) + ' ' + str(features)
            text.append(search_text)
            
            metadatas.append(
                {
                    "item_id": user_id,
                    "title": data.get('title', ''),
                    "categories": data.get('categories', []),
                    "description": data.get('description', ''),
                    "new_description": data.get('new_description', ''),
                    "price": data.get('price', 0),
                    "features": data.get('features', []),
                }
            )
        if self.verbose:
            print(f"Load Data Success, Total Data Number: {len(text)}.")

        return metadatas, text

    def load_embeddings(self, model_name):
        # Use CPU only as requested
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
        )

    def create_db(self, metadatas: List[Dict], text: List[str]):
        self.db = FAISS.from_texts(
            texts=text,
            embedding=self.embeddings,
            metadatas=metadatas,
            ids=range(len(text))
        )
        # Create directory if it doesn't exist
        os.makedirs(self.db_path, exist_ok=True)
        self.db.save_local(self.db_path)
        if self.verbose:
            print(f"Database created and saved to: {self.db_path}")

    def load_local_db(self):
        self.db = FAISS.load_local(
            folder_path=self.db_path,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )

    @tenacity.retry(reraise=True, stop=tenacity.stop_after_attempt(5),
                    wait=tenacity.wait_exponential(multiplier=1, min=10, max=120))
    def search_retriever(self, query: str):
        retriever_result = self.retriever.get_relevant_documents(query)
        if retriever_result is None:
            return None
        if len(retriever_result) == 0:
            return None
        return retriever_result


if __name__ == '__main__':
    myDB = ItemVector(
        db_path=f"path_to_local_item_database",
        model_name="bge-m3",
        llm=None,
        data_path=f"updated_item_profile.json",
        force_create=False,
        use_multi_query=False,
    )
    myDB.retriever.search_kwargs = {"k": 5}
    result = myDB.search_retriever("")
    for i in result:
        print(i.metadata)
    print(result[0].page_content)
    print(result[0].metadata['item_id'])