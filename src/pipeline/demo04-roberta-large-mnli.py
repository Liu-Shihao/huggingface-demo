from transformers import pipeline
"""
用 RoBERTa 做自然语言推理
"""

classifier = pipeline('zero-shot-classification', model='roberta-large-mnli')
sequence_to_classify = "one day I will see the world"
candidate_labels = ['travel', 'cooking', 'dancing']
results = classifier(sequence_to_classify, candidate_labels)
for result in results:
    print(result)
