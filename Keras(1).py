from sklearn.datasets import load_breast_cancer as load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score


x_train, x_text, y_train, y_test = train_test_split(scale(load().data), load().target, test_size=0.2, shuffle=True)

model = Sequential()
model.add(Dense(units=1, input_dim=x_train.shape[1], activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics='accuracy')
model.fit(x_train, y_train, epochs=20)

loss, accuracy = model.evaluate(x_text, y_test)
print(loss, accuracy)

predictions = model.predict_classes(x_text)
print(accuracy_score(predictions, y_test))
####################################################################

# from sklearn.datasets import load_boston as load
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import scale
# from keras.models import Sequential
# from keras.layers import Dense
# from sklearn.metrics import mean_absolute_error
#
#
# x_train, x_text, y_train, y_test = train_test_split(scale(load().data), load().target, test_size=0.2, shuffle=True)
#
# model = Sequential()
# model.add(Dense(units=1, input_dim=x_train.shape[1]))
# model.compile(loss='mse', optimizer='sgd')
# model.fit(x_train, y_train, epochs=20)
#
# predictions = model.predict(x_text)
# print(mean_absolute_error(predictions, y_test))
