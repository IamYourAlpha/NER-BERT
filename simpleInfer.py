# author: Intisar Chy.
# purpos: FA assignment.
# torch stuff.
import torch
from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers.models.roberta.modeling_roberta import RobertaForTokenClassification


ALL_TAGS = ['0', 'O', 'ORG-POL', 'ORG', 'PRODUCT', 'ORG-OTH', 'EVENT', 'PERSON', 'GPE', 'FAC']
label2idx = {0: '0', 1: 'O', 2: 'ORG-POL', 3: 'ORG', 4: 'PRODUCT', 5: 'ORG-OTH', 6: 'EVENT', 7: 'PERSON', 8: 'GPE', 9: 'FAC'} 
idx2label = {'0': 0, 'O': 1, 'ORG-POL': 2, 'ORG': 3, 'PRODUCT': 4, 'ORG-OTH': 5, 'EVENT': 6, 'PERSON': 7, 'GPE': 8, 'FAC': 9}
input_ =  ["SPRiNGSと最も仲の良いライバルグループ。"]
           #, "ライターの兵庫慎司は普通にアイドルポップスとして出すと売れず、無理にバンドとコラボレーションさせるのも先例からして上手くいかない、それならロックミュージシャンと制作すればいいということになったのではないかとしている。"]

def getModel(path):
    ''' getting the finetuned model '''
    config = AutoConfig.from_pretrained(
        path,
        num_labels=10,
        id2label=idx2label,
        label2id=label2idx
    )
    model = (
        RobertaForTokenClassification.from_pretrained(
            path, 
            config=config).to("cuda")
        )
    
    return model


# def getPairs():
    # {
    #     "curid": "2415752",
    #     "text": "ライターの兵庫慎司は普通にアイドルポップスとして出すと売れず、無理にバンドとコラボレーションさせるのも先例からして上手くいかない、それならロックミュージシャンと制作すればいいということになったのではないかとしている。",
    #     "entities": [
    #         {
    #             "name": "兵庫慎司",
    #             "span": [
    #                 5,
    #                 9
    #             ],
    #             "type": "人名"
    #         }
    #     ]
    # }
tokenizer =  AutoTokenizer.from_pretrained("xlm-roberta-base")
# print (tokenizer.model_max_length)
tokenizedStuffs = tokenizer(input_, truncation=True, return_tensors="pt")
# print (len(tokenizedStuffs["input_ids"][0]))
tokenizedSentence = tokenizer.convert_ids_to_tokens(tokenizedStuffs["input_ids"][0])

ids = tokenizedStuffs["input_ids"].cuda()
attnMask = tokenizedStuffs["attention_mask"].cuda()

model = getModel(path="./ckpts/best/").cuda()
outputs = model(ids, attnMask)
flat = outputs[0].view(-1, 10)
predIDX = torch.argmax(flat, dim=1)
predTags = [idx2label[i] for i in predIDX.cpu().numpy()]
print (predTags)
zipped = zip(tokenizedSentence, predTags)
for i, j in zipped:
    print (f"{i} --> {j}")