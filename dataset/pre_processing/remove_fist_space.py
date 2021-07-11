import pandas as pd

data = pd.read_csv('Emotion_Classifier/dataset/emotion_dataset_remove_space.tsv', sep = '\t')

for i in range(len(data)):
    if data.iloc[i][0][0] == ' ':
        data.iloc[i][0] = data.iloc[i][0][1:]

data.to_csv('Emotion_Classifier/dataset/emotion_dataset_remove_space.tsv', sep='\t', index=False)

print(type(data))