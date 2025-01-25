# author: Intisar Chy.
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
# torch stuff.
import torch
from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers.models.roberta.modeling_roberta import RobertaForTokenClassification

import os

# PATH = "ckpts/exp01-split80_20-epoch01-adamW/"
idx2label = {0: '0', 1: 'O', 2: 'ORG-POL', 3: 'ORG', 4: 'PRODUCT', 5: 'ORG-OTH', 6: 'EVENT', 7: 'PERSON', 8: 'GPE', 9: 'FAC'} 
label2idx = {'0': 0, 'O': 1, 'ORG-POL': 2, 'ORG': 3, 'PRODUCT': 4, 'ORG-OTH': 5, 'EVENT': 6, 'PERSON': 7, 'GPE': 8, 'FAC': 9}
# input_ =  ["SPRiNGSと最も仲の良いライバルグループ。"]
           #, "ライターの兵庫慎司は普通にアイドルポップスとして出すと売れず、無理にバンドとコラボレーションさせるのも先例からして上手くいかない、それならロックミュージシャンと制作すればいいということになったのではないかとしている。"]

PATH = os.getenv("CKPT_PATH")
app = FastAPI()
    
@app.post("/predict", response_class=JSONResponse)
async def predict(request: Request):
    data = await request.json()
    input_ = data.get("text")
    
    if not input_:
        return JSONResponse({"error": "you did not provide any input"})
    
    tokenizer =  AutoTokenizer.from_pretrained("xlm-roberta-base")
    tokenizedStuffs = tokenizer(input_, truncation=True, return_tensors="pt")
    tokenizedSentence = tokenizer.convert_ids_to_tokens(tokenizedStuffs["input_ids"][0])

    ids = tokenizedStuffs["input_ids"].cuda()
    attnMask = tokenizedStuffs["attention_mask"].cuda()
    model = getModel().cuda()
    outputs = model(ids, attnMask)
    flat = outputs[0].view(-1, 10)
    predIDX = torch.argmax(flat, dim=1)
    predTags = [idx2label[i] for i in predIDX.cpu().numpy()]
    zipped = zip(tokenizedSentence, predTags)
    returnJson = {i:j for i, j in zipped}
    print (returnJson)
    return JSONResponse(returnJson)

def getModel():
    ''' getting the finetuned model '''
    config = AutoConfig.from_pretrained(
        PATH,
        num_labels=10,
        id2label=idx2label,
        label2id=label2idx
    )
    model = (
        RobertaForTokenClassification.from_pretrained(
            PATH, 
            config=config).to("cuda")
        ).cuda()
    
    return model

