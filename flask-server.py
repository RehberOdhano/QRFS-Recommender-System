# Importing the required modules
from flask import Flask, make_response

import numpy as np 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)

def loadDataSet():
    # loading dataset
    df = pd.DataFrame(pd.read_csv(r'F:\CUI\QRFS-FYP\Backend Implementation\Flask-Server\dataset.csv'))
    df = df.loc[:,["CATEGORY", "COMPLAINT"]]
    return df

def trainLSVCModelAndGetPrediction(complaint):
    df = loadDataSet();
    # Because the computation is time consuming (in terms of CPU), the data was sampled
    df2 = df.sample(100, random_state=1, replace=True).copy()
    # Create a new column 'category_id' with encoded categories 
    df2['category_id'] = df2['CATEGORY'].factorize()[0]
    category_id_df = df2[['CATEGORY', 'category_id']].drop_duplicates()
    
    # Dictionaries for future use
    category_to_id = dict(category_id_df.values)
    id_to_category = dict(category_id_df[['category_id', 'CATEGORY']].values)
    
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, ngram_range=(1, 2), stop_words='english')
    # We transform each complaint into a vector
    features = tfidf.fit_transform(df2.COMPLAINT).toarray()
    labels = df2.category_id
    
    N = 3
    for CATEGORY, category_id in sorted(category_to_id.items()):
        features_chi2 = chi2(features, labels == category_id)
        indices = np.argsort(features_chi2[0])
        feature_names = np.array(tfidf.get_feature_names_out())[indices]
        unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
        bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
        # print("n==> %s:" %(CATEGORY))
        # print("  * Most Correlated Unigrams are: %s" %(', '.join(unigrams[-N:])))
        # print("  * Most Correlated Bigrams are: %s" %(', '.join(bigrams[-N:])))
    
    # splitting the dataset 
    # Column ‘Complaint’ will be our X or the input and the Category is out Y or the output.
    X = df2['COMPLAINT'] # Collection of documents
    y = df2['CATEGORY'] # Target or the labels we want to predict
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 0)

    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df2.index, test_size=0.25, random_state=1)
    model = LinearSVC()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 0)
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, ngram_range=(1, 2), stop_words='english')
    fitted_vectorizer = tfidf.fit(X_train)
    tfidf_vectorizer_vectors = fitted_vectorizer.transform(X_train)
    model = LinearSVC().fit(tfidf_vectorizer_vectors, y_train)
    res = model.predict(fitted_vectorizer.transform([complaint]))
    
    return res

# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.
@app.route('/')
# ‘/’ URL is bound with greetings() function.
def greetings():
	return 'WELCOME TO OUR WORLD - QRFS'

@app.route('/predict/<complaint>', methods=['POST'])
def predict(complaint):
    prediction = trainLSVCModelAndGetPrediction(complaint);
    return make_response(prediction[0], 200)

# main driver function
if __name__ == '__main__':
	# run() method of Flask class runs the application
	# on the local development server.
	app.run(debug=True)
