from transformers import pipeline
"""
使用 Fill-Mask 管道进行推理

https://huggingface.co/bert-base-uncased?text=Paris+is+the+%5BMASK%5D+of+France

Model	#params	Language
bert-base-uncased	110M	English
bert-large-uncased	340M	English
bert-base-cased	110M	English
bert-large-cased	340M	English
bert-base-chinese	110M	Chinese
bert-base-multilingual-cased	110M	Multiple
bert-large-uncased-whole-word-masking	340M	English
bert-large-cased-whole-word-masking	340M	English
"""
unmasker = pipeline('fill-mask', model='bert-base-chinese')
results = unmasker("北京市[MASK]的首都。")
for result in results:
    print(f"sequence: {result['sequence']}, with score: {round(result['score'], 4)}")