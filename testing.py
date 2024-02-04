
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# read training data and categeory

with open("maths.txt","r")as file:
    maths_file=file.read()
    
with open("philosophy.txt","r")as file:
    philosophy_file=file.read()
    
    
with open("medicine.txt","r")as file:
    medicine_file=file.read()
    



# Training data
training_data = [
    (maths_file, "maths"),
    (philosophy_file, "philosophy"),
    (medicine_file, "medicine"),
    # Add more examples with corresponding categories
]

# Create a spaCy language model
nlp = spacy.load('en_core_web_sm')

# Tokenize and lemmatize the training data
X_train = [' '.join([token.lemma_ for token in nlp(sentence)]) for sentence, _ in training_data]
y_train = [category for _, category in training_data]

# Create a text classification model pipeline
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(X_train, y_train)


#reading user file data
file_path = input("Enter the path to the file: ")

try:
    with open(file_path, 'r') as file:
        content = file.read()
except FileNotFoundError:
    print(f"File not found at path: {file_path}")
except Exception as e:
    print(f"Error reading file: {e}")

#processing the result and returuing the outpuit

processed_sent=" ".join([token.lemma_ for token in nlp(content)])
prediction_file=model.predict([processed_sent])[0]
print(f"the document belong from the categeory of :{prediction_file}")



