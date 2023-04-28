from transformers import pipeline
'''
给定一段部分词语被遮盖掉 (masked) 的文本，使用预训练模型来预测能够填充这些位置的词语。


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

pipeline 自动选择了预训练好的 distilroberta-base 模型来完成任务。
'''
unmasker = pipeline("fill-mask")
results = unmasker("This course will teach you all about <mask> models.", top_k=2)
print(results)

unmasker = pipeline('fill-mask', model='bert-base-chinese')
results = unmasker("北京是[MASK]的首都。")
for result in results:
    print(f"sequence: {result['sequence']}, with score: {round(result['score'], 4)}")