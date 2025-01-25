# author: Intisar Chy.
import json
import logging
import sys
import os
# torch stuff.
import torch
from transformers import AutoConfig
from transformers import AutoTokenizer
from dataloader import myDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from transformers.models.roberta.modeling_roberta import RobertaForTokenClassification

# for eval.
from sklearn.metrics import accuracy_score, classification_report
from utils.basicUtils import AverageMeter, accuracy

import argparse


# my loggers.
log_format = "%(asctime)s;  %(levelname)s;  %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
os.makedirs("./logs", exist_ok=True)
saveLog = os.path.join("./logs", 'faBert.log')
open(saveLog, "w")
fh = logging.FileHandler(saveLog)
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)



class MyTrainer(object):
    def __init__(self):
        self.device = "cuda"
        self.savePath = os.path.join("./ckpts", args.expName)
        os.makedirs(self.savePath, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        self.dataset = myDataset(dataPath="./datasets/ner.json", tokenizer=tokenizer)
        self.index2tag = self.dataset.index2tag
        self.tags2index = self.dataset.tags2index
        self.NERtags = self.dataset.NERTags
        # print (self.tags2index, self.index2tag)
        logging.info(f"All the classes: {self.NERtags}")
        logging.info(f"index to tag: {self.index2tag}")
        logging.info(f"tag to index: {self.tags2index}")
        
        trainSize = int(0.8 * len(self.dataset))
        evalSize = len(self.dataset) - trainSize

        # Split the dataset
        trainDataset, evalDataset = random_split(self.dataset, [trainSize, evalSize])
        logging.info(f"total number of samples: {len(self.dataset)}")
        logging.info(f"total number of train samples: {len(trainDataset)}")
        logging.info(f"total number of eval samples: {len(evalDataset)}")
        # Best practice would be to have a config file where I set everything.
        # due to my  limited compute resource I set low batch size.
        self.trainingLoader = DataLoader(trainDataset, batch_size=16, shuffle=True, num_workers=0)
        self.testingLoader = DataLoader(evalDataset, batch_sampler=16, shuffle=False, num_workers=0)
        
        
        # get model only for testing.
        # self.model = self._getMyModel()
 
        # get the base model.
        if (args.trainMode): # if in training mode I start with base model.
            logging.info(f"In training mode, so loading the base model")
            self.model = self._getModel()
        else:
            logging.info(f"In eval. model. Loading finetuned model from: {self.savePath}")
            self.model = self._getMyModel()
        coun = sum(p.numel() for p in  self.model.parameters() if p.requires_grad==True)/1e6
        logging.info(f"Model size: {coun} M")
        os.makedirs(self.savePath, exist_ok=True)


    def _getModel(self):
        ''' getting the base model '''
        xlmr_config = AutoConfig.from_pretrained(
            "xlm-roberta-base",
            num_labels=len(self.NERtags),
            id2label=self.index2tag,
            label2id=self.tags2index
        )
        model = (
            RobertaForTokenClassification.from_pretrained(
                "xlm-roberta-base", 
                config=xlmr_config).to("cuda")
            )
        return model

    
    def _getMyModel(self):
        ''' getting the finetuned model '''
        
        try:
            config = AutoConfig.from_pretrained(
                self.savePath,
                num_labels=len(self.NERtags),
                id2label=self.index2tag,
                label2id=self.tags2index
            )
            model = (
                RobertaForTokenClassification.from_pretrained(
                    self.savePath, 
                    config=config).to("cuda")
                )
        except Exception as e:
            logging.error(f"The path {self.savePath} does not exists or the finetuned model does not exists")
            exit(0)
        
        return model


    def trainOneEpoch(self, epoch, optimizer):
        ''' one epoch trainer '''
        losses     = AverageMeter()
        self.model.train()
        totLoss = 0.0
        iterCounter = 0
        predList = list()
        targList = list()
        for idx, batch in enumerate(self.trainingLoader, start=1):
            inputIDS = batch['inputIDS'].cuda()
            attnMask = batch['attnMask'].cuda()
            targets = batch['targets'].cuda()
     
            outputs = self.model(input_ids=inputIDS, attention_mask=attnMask, labels=targets)
            loss, Zs = outputs.loss, outputs.logits
            
            iterCounter += 1
            totLoss += loss.item()
            
            # if (idx % 1 == 0):
            #     lossSoFar = totLoss/iterCounter
            #     print(f"Loss: {lossSoFar}")
             
            # print (targets.shape)
            targetsFlat = targets.view(-1)   
            logitPredFlat = Zs.view(-1, 10) 
            logitPredSingle = torch.argmax(logitPredFlat, axis=1)
            indexToConsider = attnMask.view(-1) == 1 # active accuracy is also of shape (batch_size * seq_len,)            
            validTargets = torch.masked_select(targetsFlat, indexToConsider)
            validPreds = torch.masked_select(logitPredSingle, indexToConsider)
            # print (validTargets.shape, validPreds.shape)
            losses.update(loss.item(), inputIDS.size(0))
                     
            predList.extend(validPreds.cpu().tolist())
            targList.extend(validTargets.cpu().tolist())
            
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(
                parameters=self.model.parameters(), max_norm=10
            )
            
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (idx % 100 == 0):
                log_str = f'Epoch[{epoch}]:[{idx:03}/{len(self.trainingLoader):03}] ' \
                    f'loss: {losses.val:.4f}, loss_avg: {losses.avg:.4f}' 
                logging.info(log_str)
              

        # acc = accuracy_score(targList, predList)
        # print(f"Training loss epoch: {lossAvg}")
        # print(f"Training accuracy epoch: {acc * 100}")
        # return lossAvg
    
    # @staticmethod
    # def simpleInfer(dummyInput):
    #     pass
    
    def test(self):
        losses = AverageMeter()
        self.model.eval()
        totLoss = 0.0
        iterCounter = 0
        predList = list()
        targList = list()
        
        with torch.no_grad():
            for _, batch in enumerate(self.trainingLoader, start=1):
                inputIDS = batch['inputIDS'].cuda()
                attnMask = batch['attnMask'].cuda()
                targets = batch['targets'].cuda()

                outputs = self.model(input_ids=inputIDS, attention_mask=attnMask, labels=targets)
                loss, Zs = outputs.loss, outputs.logits
                
                iterCounter += 1
                totLoss += loss.item()
                
                targetsFlat = targets.view(-1)   
                logitPredFlat = Zs.view(-1, 10) 
                logitPredSingle = torch.argmax(logitPredFlat, axis=1)
                indexToConsider = attnMask.view(-1) == 1 # active accuracy is also of shape (batch_size * seq_len,)            
                validTargets = torch.masked_select(targetsFlat, indexToConsider)
                validPreds = torch.masked_select(logitPredSingle, indexToConsider)
                losses.update(loss.item(), inputIDS.size(0))
                        
                predList.extend(validPreds.cpu().tolist())
                targList.extend(validTargets.cpu().tolist())
                
                   
        log_str = f'loss: {losses.val:.4f}, loss_avg: {losses.avg:.4f}' 
        logging.info(log_str)
        
        
        namedTarget = [self.index2tag[idx] for idx in targList]
        namedPred = [self.index2tag[idx] for idx in predList]
        acc = accuracy_score(targList, predList)
        return acc, namedTarget, namedPred
 
    
    def trainEpoch(self):
        optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=0.00005, weight_decay=1e-4)
        bestSoFar = -999
        for i in range(1):
            self.trainOneEpoch(i, optimizer)
            currAcc, _, _ = self.test()
            logging.info(f"Current eval. acc.: {round(currAcc * 100)}%")
            if (currAcc > bestSoFar):
                bestSoFar = currAcc
                self.model.save_pretrained(self.savePath) # not saving all, over-rididng.
                logging.info(f"Saving best checkpoint..")
            
            
    def calcACC(self):
        # load the checkpoint first, as I will call this
        # without training or testing at times.
        _, tar, pred = self.test()
        fullPreds = classification_report(tar, pred)
        logging.info(f"\n{fullPreds}")
        
        
#####################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--trainMode', action='store_true', help='enable training mode')
parser.add_argument('--evalMode', action='store_true', help='enable evaluation mode')
parser.add_argument('--expName', type=str, default="fa-bert", help='exp. name')

args = parser.parse_args()
#####################################################################################

obj = MyTrainer()           
if (args.trainMode):
    obj.trainEpoch()
    obj.calcACC()
else:
    obj.calcACC() # it will read the checkpoint from folder ckpts/best/