from sklearn.ensemble import GradientBoostingClassifier
import time
import re
import matplotlib.pyplot as plt


def i_feel_lucky_xgboost_training(train_df, test_df, features, target, name,
                                  n_estimators=80, max_depth=4, learning_rate=0.05):
    x_train = train_df[features]
    y_train = train_df[target]
    x_test = test_df[features]
    y_test = test_df[target]

    xgb_clf = GradientBoostingClassifier()
    
    start = time.time()
    xgb_clf.fit(x_train, y_train.values.ravel())
    end = time.time()
    clf_name = name
    test_df[clf_name] = xgb_clf.predict(x_test)#[:, 1]
    return xgb_clf, clf_name

def clean_text(text, stop_words):
    #remove punctuation and numbers
    text = re.sub("[^a-zA-Z]", ' ', text)
    #lowercase
    text = text.lower().split()
    #remove stopwords
    text = [w for w in text if w not in stop_words]
    #join to 1 string
    text = " ".join(text)
    return text


def plot_model_param(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()