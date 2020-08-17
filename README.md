# categorical_embeddings_pokemon_dataset
Tabular data is the most commonly used type of data in industry.  
This repo covers some key concepts from applying neural networks to tabular data, in particular the idea of creating embeddings for categorical variables.  
There are few ways to convert categorical data but in the situation that an attribute has a lot of unique values or allow for relationships between categories to be captured.   
Dataset was taken from https://www.kaggle.com/abcsds/pokemon/data#  
To use Tensorboard navigate to the directory (type or gen) and run `tensorboard --logdir=./`  
Evaluations : `accuracy :  0.92 precision :  0.50 recall :  0.80 auc :  0.97`  
Results can be improved using different NN architectures


# Requirements
pandas  
numpy  
scikit-learn  
seaborn  
matplotlib 