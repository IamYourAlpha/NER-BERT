import requests

text = "かつては東京都三鷹市にオフィスを構えていたが、大月のキングレコード職を機に事業を停止。"
response = requests.post("http://localhost:8000/predict", json={"text": text})


VALID_TAGS = ['ORG-POL', 'ORG', 'PRODUCT', 'ORG-OTH', 'EVENT', 'PERSON', 'GPE', 'FAC']
INVALID_TAGS = ['0', 'O']
BAD_WORDS = ['▁', '<s>', '</s>']
output = {}
fullText = ""
if response.status_code == 200:
    json_ = response.json()
    tokens, tags, words, wordTags = [], [], [], []
    for k, v in json_.items():
        tokens.append(k)
        tags.append(v)
        
    current_word = ""
    current_tag = None
    for i, token in enumerate(tokens):
        if token not in BAD_WORDS:
            fullText += token  # for debug.

        if i > 0 and tags[i] == tags[i - 1]:  # Continue the current word
            current_word += token
        else:  #end of word.
            if current_word:
                current_word = current_word.strip()
                current_word = current_word.strip("▁")
                output[current_word] = current_tag
                current_word = ""

            if token not in BAD_WORDS:  # starting a new word
                current_word = token
                current_tag = tags[i]

    if current_word:  # add the last word
        output[current_word] = current_tag
    # out.write(f"success: {output}")
                
else:
    print(f"Error: {response.status_code}")

# ideal output
# {'かつては': 'O', '東京都三鷹市': 'LOC', 'にオフィスを構えていたが、': 'O', '大月のキングレコード': 'ORG', '職機事業停止。': 'O'}