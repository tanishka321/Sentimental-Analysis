import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, classification_report
from scikitplot.metrics import plot_confusion_matrix
from matplotlib import rcParams

# Load training and validation datasets
df_train = pd.read_csv("train.txt", delimiter=';', names=['text', 'label'])
df_val = pd.read_csv("val.txt", delimiter=';', names=['text', 'label'])

# Combine training and validation datasets
df = pd.concat([df_train, df_val])
df.reset_index(inplace=True, drop=True)

# Display shape and sample of the DataFrame
print("Shape of the DataFrame:", df.shape)
print(df.sample(5))

# Function to encode labels
def custom_encoder(df):
    df.replace(to_replace="surprise", value=1, inplace=True)
    df.replace(to_replace="love", value=1, inplace=True)
    df.replace(to_replace="joy", value=1, inplace=True)
    df.replace(to_replace="fear", value=0, inplace=True)
    df.replace(to_replace="anger", value=0, inplace=True)
    df.replace(to_replace="sadness", value=0, inplace=True)

# Apply custom encoding to labels
custom_encoder(df['label'])

# Function to preprocess text data
lm = WordNetLemmatizer()
def text_transformation(df_col):
    corpus = []
    for item in df_col:
        new_item = re.sub('[^a-zA-Z]', ' ', str(item))
        new_item = new_item.lower()
        new_item = new_item.split()
        new_item = [lm.lemmatize(word) for word in new_item if word not in set(stopwords.words('english'))]
        corpus.append(' '.join(str(x) for x in new_item))
    return corpus

# Transform text data
corpus = text_transformation(df['text'])

# Generate word cloud
rcParams['figure.figsize'] = 20, 8
word_cloud = " ".join(corpus)
wordcloud = WordCloud(width=1000, height=500, background_color='white', min_font_size=10).generate(word_cloud)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

# Convert text data into vectors
cv = CountVectorizer(ngram_range=(1, 2))
traindata = cv.fit_transform(corpus)
X = traindata
y = df.label

# Define hyperparameters for grid search
parameters = {
    'max_features': ['auto', 'sqrt'],
    'n_estimators': [500, 1000, 1500],
    'max_depth': [5, 10, None],
    'min_samples_split': [5, 10, 15],
    'min_samples_leaf': [1, 2, 5, 10],
    'bootstrap': [True, False]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(RandomForestClassifier(), parameters, cv=5, return_train_score=True, n_jobs=-1)
grid_search.fit(X, y)
best_params = grid_search.best_params_

# Print grid search results
for i in range(len(grid_search.cv_results_['params'])):
    print('Parameters:', grid_search.cv_results_['params'][i])
    print('Mean Test Score:', grid_search.cv_results_['mean_test_score'][i])
    print('Rank:', grid_search.cv_results_['rank_test_score'][i])

# Train Random Forest Classifier with the best parameters
rfc = RandomForestClassifier(
    max_features=best_params['max_features'],
    max_depth=best_params['max_depth'],
    n_estimators=best_params['n_estimators'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    bootstrap=best_params['bootstrap']
)
rfc.fit(X, y)

# Load test dataset
test_df = pd.read_csv('test.txt', delimiter=';', names=['text', 'label'])
X_test, y_test = test_df.text, test_df.label

# Encode labels in the test set
custom_encoder(y_test)

# Preprocess test text data
test_corpus = text_transformation(X_test)
testdata = cv.transform(test_corpus)

# Make predictions on the test set
predictions = rfc.predict(testdata)

# Plot confusion matrix
rcParams['figure.figsize'] = 10, 5
plot_confusion_matrix(y_test, predictions)
plt.show()

# Print evaluation metrics
acc_score = accuracy_score(y_test, predictions)
pre_score = precision_score(y_test, predictions)
rec_score = recall_score(y_test, predictions)
print('Accuracy Score:', acc_score)
print('Precision Score:', pre_score)
print('Recall Score:', rec_score)
print("-" * 50)
print(classification_report(y_test, predictions))
