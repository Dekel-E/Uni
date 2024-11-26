import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Setting the values for the upcoming matplotlib graphs.
plt.rcParams["figure.figsize"] = (12, 12)

# Setting the required train and test sizes.
train_size = 700
test_size = 300


# Finding which distribution will be used for each of the points.
def generate_distrib(size):
    return sorted(np.random.randint(1, 4, size=size))


train_distrib = generate_distrib(train_size)
test_distrib = generate_distrib(test_size)

# Creating the parameters for the multivariate normal distributions provided in the question.
cov = np.eye(2)
mu_1 = np.array([-1, 1], dtype='float').T
mu_2 = np.array([-2.5, 2.5], dtype='float').T
mu_3 = np.array([-4.5, 4.5], dtype='float').T  # Version 2
mu_array = [mu_1, mu_2, mu_3]


# Generating all of the samples for the train and test sets.
def generate_set(size, distrib):
    X = np.zeros(size)
    Y = np.zeros(size)
    for i in range(size):
        # Each distribution in train_distrib is from 1 to 3, while the corresponding mean in mu_array is 1 less.
        X[i], Y[i] = np.random.multivariate_normal(mu_array[distrib[i] - 1], cov, 1)[0]
    return pd.DataFrame(list(zip(X, Y, distrib)), columns=['x', 'y', 'distrib'])


train_set = generate_set(train_size, train_distrib)
test_set = generate_set(test_size, test_distrib)

# Plotting the points.
distribs = [1, 2, 3]
colors = ['r', 'g', 'b']


def plot(dataset, title, x_col_name='x', y_col_name='y'):
    # Adding scatterplots
    for distrib, color in zip(distribs, colors):
        # Filtering by the current label.
        plot_data = dataset[dataset['distrib'] == distrib]
        # Plotting the scatter plot.
        plt.scatter(x=plot_data['x'], y=plot_data['y'], label=distrib, color=color, s=50)

    # Setting the basic name parameters.
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)

    # Specifying the ticks for the axis.
    plt.xticks(np.arange(dataset['x'].min(), dataset['x'].max(), 1))
    plt.yticks(np.arange(dataset['y'].min(), dataset['y'].max(), 1))
    plt.show()


# Plot training and test sets
plot(train_set, 'Training set')
plot(test_set, 'Test set')

# Training the kNN classifier using k=1
k = 1
x_cols = ['x', 'y']
y_cols = 'distrib'

# Instantiate and fit the classifier
model = KNeighborsClassifier(n_neighbors=k)
model.fit(train_set[x_cols], train_set[y_cols])

# Predicting
y_train_true = train_distrib
y_test_true = test_distrib
y_train_pred = model.predict(X=train_set[x_cols])
y_test_pred = model.predict(X=test_set[x_cols])

# Finding the accuracy of the model
error_rate_train = np.sum(y_train_true != y_train_pred) / train_size
error_rate_test = np.sum(y_test_true != y_test_pred) / test_size

print(f'Classification Error Rate for the train set is: {error_rate_train}, or {error_rate_train * 100:.2f}%')
print(f'Classification Error Rate for the test set is: {error_rate_test:.2f}, or {error_rate_test * 100:.2f}%')

# Finding the errors for each of the k values from 1 to 20.
k_arr = np.arange(1, 21, 1)
train_errors = []
test_errors = []

for k in k_arr:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(train_set[x_cols], train_set[y_cols])

    y_train_pred = model.predict(X=train_set[x_cols])
    y_test_pred = model.predict(X=test_set[x_cols])

    train_errors.append(np.sum(y_train_true != y_train_pred) / train_size)
    test_errors.append(np.sum(y_test_true != y_test_pred) / test_size)

# Plotting errors vs k
plt.figure()
plt.plot(k_arr, train_errors, label="Train Errors")
plt.plot(k_arr, test_errors, label="Test Errors")
plt.xlabel('K')
plt.ylabel('Error')
plt.grid(True)
plt.legend()
plt.xticks(k_arr)
plt.show()

# Analyzing effect of training set size
k = 10  # Fix k=10 for this analysis
train_size_arr = np.arange(10, 41, 5)  # From 10 to 40 with step 5
test_size = 100  # New test size for this analysis

# Generate new test set for this analysis
test_distrib = generate_distrib(test_size)
test_set = generate_set(test_size, test_distrib)

train_errors = []
test_errors = []

for train_size in train_size_arr:
    # Generate new training data for each size
    train_distrib = generate_distrib(train_size)
    train_set = generate_set(train_size, train_distrib)

    # Train model
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(train_set[x_cols], train_set[y_cols])

    # Predict and calculate training error
    y_train_pred = model.predict(X=train_set[x_cols])
    train_error = np.sum(train_set['distrib'] != y_train_pred) / train_size
    train_errors.append(train_error)

    # Predict and calculate test error
    y_test_pred = model.predict(X=test_set[x_cols])
    test_error = np.sum(test_set['distrib'] != y_test_pred) / test_size
    test_errors.append(test_error)

# Plot errors vs training set size
plt.figure()
plt.plot(train_size_arr, train_errors, label='Train Errors', color='blue')
plt.plot(train_size_arr, test_errors, label='Test Errors', color='orange')
plt.xlabel('Train set Size')
plt.ylabel('Classification error')
plt.grid(True)
plt.legend()
plt.xticks(train_size_arr)
plt.show()