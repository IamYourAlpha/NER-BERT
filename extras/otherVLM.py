# Load model directly
from transformers import AutoTokenizer, AutoModelForPreTraining, AutoConfig
# need fugashi, unidic_lite --> pip install unidic-lite, fugashi
tokenizer = AutoTokenizer.from_pretrained("tohoku-nlp/bert-base-japanese-v3")
config = AutoConfig.from_pretrained("tohoku-nlp/bert-base-japanese-v3", 
                num_labels=2,
                id2label={0: 'OK', 1: 'debug'},
                label2id={'OK':0, 'debug': 1})
model = AutoModelForPreTraining.from_pretrained("tohoku-nlp/bert-base-japanese-v3", config=config)
print (config)
print (model)

