# simple dataloader to prepare dataset.

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import json

from transformers import AutoTokenizer


class myDataset(Dataset):
    def __init__(self, dataPath=None, tokenizer=None):
        """my simple dataloader"""
        with open(dataPath) as jsonData:
            rawData = json.load(jsonData)
        self.rawJson = rawData
        self.tokenizer = tokenizer
        self.dataFrame = self._createDataFrame()
        self.NERTags = ['0', 'O', 'ORG-POL', 'ORG', 'PRODUCT', 'ORG-OTH', 'EVENT', 'PERSON', 'GPE', 'FAC']
        self.index2tag = {k: v for k, v in enumerate(self.NERTags)}
        self.tags2index = {v: k for k, v in enumerate(self.NERTags)}
        # print (self.dataFrame.head())
        self.tokenizedFull = self._formatDataFrame(self.dataFrame)
        self.maxLen = 128 # dont need as we use default.
    
    def _createDataFrame(self):

        entity_type_mapping = {
            "地名": "GPE",
            "施設名": "FAC",
            "製品名": "PRODUCT",
            "イベント名": "EVENT",
            "人名": "PERSON",
            "法人名": "ORG",
            "政治的組織名": "ORG-POL",
            "その他の組織名": "ORG-OTH"
        }

        data = {"tokens": [], "ner_tags": []}
        for json_dat in self.rawJson:
            tokens = list(json_dat["text"])
            ner_tags = ["O"] * len(tokens)
            for ent in json_dat["entities"]:
                for i in range(ent["span"][0], ent["span"][1]):
                    ner_tags[i] = entity_type_mapping.get(ent["type"], "O")
                data["tokens"].append(tokens)
                data["ner_tags"].append(ner_tags)

        return pd.DataFrame(data)
    
    def _formatDataFrame(self, data):
        # this particular pre-processing function is from the internet.
        text = ["".join(t) for t in data["tokens"]]
        tokenized_inputs = self.tokenizer(text, padding=True)
        print (tokenized_inputs)
        labels = []
        for row_idx, label_old in enumerate(data["ner_tags"]):
            label_new = [[] for _ in tokenized_inputs.tokens(batch_index=row_idx)]  
            for char_idx, label in enumerate(label_old): 
                token_idx = tokenized_inputs.char_to_token(row_idx, char_idx)  
                if token_idx is not None:
                    label_new[token_idx].append(label) 
                    if (tokenized_inputs.tokens(batch_index=row_idx)[token_idx] == " ") and (label != 0):  
                        label_new[token_idx+1].append(label)
                        
            labels.append([max(i, default=0) for i in label_new]) 
        tokenized_inputs["labels"] = labels
        
        return tokenized_inputs
    
    def __getitem__(self, idx):
        
        ids =  self.tokenizedFull[idx].ids
        attn_mask = self.tokenizedFull[idx].attention_mask
        labels = self.tokenizedFull['labels'][idx]
        label_ids = [self.tags2index[str(label)] for label in labels]
        # some sanity check.
        # print (ids)
        # print (self.tokenizer.convert_ids_to_tokens(ids))
        # print (self.tokenizer.convert_tokens_to_ids(self.tokenizer.convert_ids_to_tokens(ids)))
 
        return {
            'inputIDS': torch.tensor(ids, dtype=torch.long),
            'attnMask': torch.tensor(attn_mask, dtype=torch.long),
            'targets': torch.tensor(label_ids, dtype=torch.long)
        }
    
    def __len__(self):
        return len(self.dataFrame)

if __name__ == "__main__":
    # for debug
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    print (tokenizer)
    with open("./datasets/ner.json") as jsonData:
        rawData = json.load(jsonData)
    dataset = myDataset(rawData, tokenizer)
    for data in dataset:
        print (data)
    
    
