from sklearn.neural_network import MLPClassifier   
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 


class MultiLayerPerceptronClassifier(BaseEstimator, ClassifierMixin):
    """ Perceptron classifier, which determines the user's intent. """
    def __init__(self, hiddien_layer=80, neuron=80):    
        """ Create a new object """
        self.feature_extraction = TfidfVectorizer()   
        self.clf = MLPClassifier(hidden_layer_sizes=(neuron,hiddien_layer),activation='relu', solver='adam', random_state=42)

    def fit(self, X, y, **kwargs):
        """ Fit the perceptron to convert sequence to intent.
        
        :param X: input texts for training.
        :param y: target intents for training.
        
        :return self
        
        """
        self.tfidf = self.feature_extraction.fit(X)
        X_train = self.tfidf.transform(X)
        self.clf.fit(X_train, y)
        return self

	def predict(self, X):
        """ Predict resulting intents by source sequences with a trained logistic regression model.
        
        :param X: source sequences.
        
        :return: resulting intents, predicted for source sequences.
        
        """
        X_train = self.tfidf.transform([X])
        return self.clf.predict(X_train)
    
