import implicit
import numpy
import pandas
import scipy

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Hyperparameters
factors = 20
regularization = 0.1
threshold = 10
n_estimators = 100
criterion = "gini"
min_samples_leaf = 2
max_features = 0.2

data = pandas.read_table("lezhin_dataset_v2_training.tsv", header=None)
print("Data loaded")

data["item"] = data[6]
data["user"] = data[7]

item_encoder = LabelEncoder()
user_encoder = LabelEncoder()
data["item_code"] = item_encoder.fit_transform(data["item"])
data["user_code"] = user_encoder.fit_transform(data["user"])
print("Code done")

train, test = train_test_split(data)

train_buy = train.loc[train[0] == 1, ["item_code", "user_code"]]
train_buy_count = train_buy.groupby(["item_code", "user_code"]).aggregate(len)
V = train_buy_count.values.astype(float)
V = numpy.where(V >= threshold, threshold, V)
I = train_buy_count.index.get_level_values("item_code").values
J = train_buy_count.index.get_level_values("user_code").values
train_matrix = scipy.sparse.coo_matrix((V, (I, J))).tocsr()

ALS = implicit.als.AlternatingLeastSquares
F = ALS(factors=factors, regularization=regularization)
F.fit(train_matrix)

mean_I = F.item_factors.mean(axis=0)
mean_U = F.user_factors.mean(axis=0)
n_I = len(item_encoder.classes_) - len(F.item_factors)
n_U = len(user_encoder.classes_) - len(F.user_factors)
I = numpy.repeat([mean_I], n_I, axis=0)
U = numpy.repeat([mean_U], n_U, axis=0)
F.item_factors = numpy.concatenate((F.item_factors, I))
F.user_factors = numpy.concatenate((F.user_factors, U))

columns_I = ["item_{}".format(i + 1) for i in range(factors)]
columns_U = ["user_{}".format(i + 1) for i in range(factors)]
item_factors = pandas.DataFrame(F.item_factors, columns=columns_I)
user_factors = pandas.DataFrame(F.user_factors, columns=columns_U)
print("Factorization done")

columns_buy = ["buy_{}".format(i + 1) for i in range(100)]
columns_factors = columns_I + columns_U

train[columns_buy] = train[list(range(10, 110))]
train = train.join(item_factors, on="item_code")
train = train.join(user_factors, on="user_code")

test[columns_buy] = test[list(range(10, 110))]
test = test.join(item_factors, on="item_code")
test = test.join(user_factors, on="user_code")

def evaluate(model, columns):
    X_train = train[columns]
    y_train = train[0]
    X_test = test[columns]
    y_test = test[0]
    model.fit(X_train, y_train)
    classes = model.classes_
    index = list(classes).index(1)
    y_probs = model.predict_proba(X_test)
    y_prob = y_probs[:, index]
    score = roc_auc_score(y_test, y_prob)
    print("{:.4f}".format(score))

model = RandomForestClassifier(n_estimators=n_estimators,
                               criterion=criterion,
                               min_samples_leaf=min_samples_leaf,
                               max_features=max_features,
                               n_jobs=8,
                               verbose=2)
columns = columns_buy + columns_factors
evaluate(model, columns)
