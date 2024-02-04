
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import csv
nlp=spacy.load("en_core_web_sm")

# read training data and categeory

with open("maths.txt","r")as file:
    maths_file=file.read()
    
with open("philosophy.txt","r")as file:
    philosophy_file=file.read()
    

    
 # core topics:
   
with open("medicine.txt","r")as file:
    medicine_file=file.read()
    
with open("legal.csv","r")as file:
    legal_data=list(csv.reader(file))
    legal_file=" ".join([" ".join(row)for row in legal_data])
    
with open("medical.csv","r")as file:
    medical_data=list(csv.reader(file))
    medical_file=" ".join([" ".join(row)for row in medical_data])
    
with open("finance.csv","r")as file:
    finance_data=list(csv.reader(file))
    finance_file=" ".join([" ".join(row)for row in finance_data])
  
with open("business.csv","r")as file:
    business_data=list(csv.reader(file))
    business_file=" ".join([" ".join(row)for row in business_data])
    
    
with open("government.txt","r")as file:
    government_file=file.read()
  
with open("news.txt","r")as file:
    news_file=file.read()
    
with open("maunals.txt","r")as file:
    manuals_file=file.read()
    
with open("research.txt","r")as file:
    research_file=file.read()
with open("creative.txt","r")as file:
    creative_file=file.read()
    

  # topics reding ends



# Training data
training_data = [
    (maths_file, "maths"),
    (philosophy_file, "philosophy"),
    (medicine_file, "medicine"),
    (legal_file, "legal document"),
    (medical_file, "medical"),
    (finance_file, "finance"),
    (business_file, "business"),
    (government_file, "government"),
    (news_file, "news"),    
    (manuals_file, "maunals"),  
    (research_file, "research"),  
    (creative_file, "creative"),  
]



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



