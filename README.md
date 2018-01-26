# NTUOSS Machine Learning and Data Science Workshop

*by [Chaitanya Joshi](https://chaitjo.github.io) for [NTU Open Source Society](http://ntuoss.com)*

---

Welcome! Today, we'll be going through a "Hello, world!" example for Machine Learning, all the way from data exploration to building viable machine learning models using the scikit-learn library for Python. By the end of the workshop, you'd have built a model to recognize handwritten digits while getting a whirlwind tour of the basic techniques and programming tools in Data Science.

***Disclaimer** - This document is only meant to serve as a reference for the attendees of the workshop. It does not cover all the concepts or implementation details discussed during the actual workshop.*

### Workshop Details:

When?: Friday, 26 January 2018. 6:30 PM - 8:30 PM.

Where?: Nanyang Technological University

Who?: NTU Open Source Society

### Questions?

Raise your hand at any time during the workshop or [email me](mailto:ckjoshi9@gmail.com) afterwards.

# Machine Learning 101

### Let's watch [a video](https://www.youtube.com/watch?v=f_uwKZIAeM0)!

Machine learning algorithms differ from standard algorithms (such as searching or sorting) in the sense that they **do not need to be explicitly programmed** to do a task. Such algorithms can *learn* how to solve the given task from *data*.

![Facial Recognition](/img/facial-recognition.jpeg)

## What kind of data?

In a general ML problem, we start with many samples of data and then try to predict properties of unknown data. If each sample is more than a single number and, for instance, a multi-dimensional entry (aka multivariate data), it is said to have several attributes or features.

For example, the [Iris flower dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set) is one of the most popular examples of a dataset with multiple features. Check out the [iris-dataset.csv](/res/iris-dataset.csv) file.

## Supervised vs Unsupervised Learning

![Supervised Learning vs Unsupervised Learning](/img/supervised-unsupervised.png)

In **supervised learning**, the data consists of a set of features as well as some additional attributes that we want to predict (i.e. it has been **labelled** beforehand). Given just the features for some new data, we now want to **classify** it into a category or perform a **regression** to predict a numerical value.

In **unsupervised learning**, the data consists of a set of inputs without any corresponding target values (i.e. **unlabelled data**). The goal in such problems is to discover groups of similar examples within the data, where it is called **clustering**.

![Labelled data](/img/labelled-data.png)

![Types of ML Algorithms](/img/ml-types.jpg)

# Task 0: Setup

## Step 1: Install Python 3 and Pip
Check [this](https://github.com/jarrettyeo/NTUOSS-PythonPipInstallation) out.

## Step 2: Use pip to install dependencies

### Windows

Run cmd as administrator, then execute:
```
pip install numpy scikit-learn matplotlib ipython
```

### Linux/Mac

Open the terminal, and execute:
```
sudo pip3 install numpy scikit-learn matplotlib ipython
```

Key in your password when prompted. You will not be able to see anything as you type your password into the console.

## Step 3: Decide on workflow

Once all dependencies are installed, we're finally ready to build our first machine learning model. You can write the code for the rest of the tutorial in whatever way you like:
1. Use IDLE
2. Use the terminal or command prompt to run a python shell, or better yet, an ipython shell. (Just type in `python` or `ipython`.)
3. Write the code in a .py file and execute it as we go along

## Step 4: Import libraries

Import the libraries you installed:
```python
import numpy as np
import sklearn
import matplotlib.pyplot as plt
```

During the course of this tutorial, we shall be importing more functions/classes from these libraries as we build our model.

# Task 1: Data Exploration

## Step 1: Loading data

The first step to about anything in data science is loading in your data. For now, don't worry about finding any data by yourself and just load in the handwritten digits dataset that comes with scikit-learn.

Paste the code in your Python shell/.py file and fill in the blanks using the **[scikit-learn toy datasets documentation](http://scikit-learn.org/stable/datasets/index.html#toy-datasets)**:

```python
# Import `datasets` from `sklearn`
from sklearn import datasets

# Load in the `digits` data
digits = datasets._____()

# Print the `digits` data
print(_____)
```

Hint: See the `load_digits()` method.

## Step 2: Gathering basic information

When you printed out the digits data, you will have noticed that it is a Python dictionary object with various keys. Lets look at what kinds of information we have availabe by printing the keys:

```python
# Get the keys of the `digits` data
print(digits.keys())
```

Let's have a closer look at the data, target and description of the dataset:

```python
# Print out the data
print(digits._____)

# Print out the target values
print(digits._____)

# Print out the description of the `digits` data
print(digits._____)
```

### What does each data item mean?

![Images to pixel values](/img/digits.png)

To store such arrays or matrices, we use an object called a numpy array. We could have used standard lists, but numpy arrays are more powerful and specifically designed for performing fast mathematical operations.

We can treat them in a similar way to lists for now, but we shall be covering them more in our next workshop.

![Arrays](/img/arrays.png)

## Step 3: Inspecting the shape: How much data do we have?

The first thing that you should know of an array is its shape. That is, the number of dimensions and items that is contained within an array. The array’s shape is a tuple of integers that specify the sizes of each dimension. In other words, if you have a 3d array like this:

```
|                                                   |
| | 00  01  02 |   | 00  01  02 |   | 00  01  02 |  |
| | 10  11  12 | , | 10  11  12 | , | 10  11  12 |  |	 				 				  
| | 20  21  22 |   | 20  21  22 |   | 20  21  22 |  |						 
|                                                   |

It will be represented as: 
    np.array([ 
                [ 
                    [00, 01, 02], 
                    [10, 11, 12], 
                    [20, 21, 22] 
                ],
                [ [00, 01, 02], [10, 11, 12], [20, 21, 22] ],
                [ [00, 01, 02], [10, 11, 12], [20, 21, 22] ]
             ])
             
Since its a 3x3x3 array, its shape is represented by the tuple: (3,3,3) 
```

Refer to the **[Numpy array documentation](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.shape.html)** and find the shape of the data, target values and images.

```python
# Isolate the `digits` data
digits_data = digits.data

# Inspect the shape
print(digits_data.____)

# Isolate the target values with `target`
digits_target = digits.target

# Inspect the shape
print(digits_target.____)

# Print the number of unique labels
number_digits = len(np.unique(digits.target))

# Isolate the `images`
digits_images = digits.images

# Inspect the shape
print(digits_images.____)
```

## Step 4: Visualizing data

Data visualization often helps us get a better idea about our data and come up with initial intuitions about how to solve the given problem. Humans tend to think visually, after all!  The matplotlib library allows us to chart, plot and visualize datasets to gain more insight into how we should approach any data science task. 

We shall be covering data visualization in depth next week, so just copy paste the following sections of code into Python for now.

```python
# Figure size (width, height) in inches
fig = plt.figure(figsize=(6, 6))

# Adjust the subplots
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# For each of the 64 images
for i in range(64):
    # Initialize the subplots: add a subplot in the grid of 8 by 8, at the i+1-th position
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    # Display an image at the i-th position
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
    # label the image with the target value
    ax.text(0, 7, str(digits.target[i]))

# Show the plot
plt.show()
```

We took the first 64 images in our dataset and plotted them on an 8x8 grid. You should see something like this:

![](/img/plot1.png)

We can also visualize the target labels with some images using the following code:

```python
# Join the images and target labels in a list
images_and_labels = list(zip(digits.images, digits.target))

# for every element in the list
for index, (image, label) in enumerate(images_and_labels[:8]):
    # initialize a subplot of 2X4 at the i+1-th position
    plt.subplot(2, 4, index + 1)
    # Don't plot any axes
    plt.axis('off')
    # Display images in all subplots
    plt.imshow(image, cmap=plt.cm.gray_r,interpolation='nearest')
    # Add a title to each subplot
    plt.title('Training: ' + str(label))

# Show the plot
plt.show()
```

This should render the following visualization:

![](/img/plot2.png)

We now have a very good idea of the data we're working with. But is there any other way to visualize all the digits together? Maybe on a graph?

### Dimensionality reduction

As the digits data set contains 64 features, this might prove to be a challenging task. You can imagine that it’s very hard to understand the structure and keep the overview of the digits data. In such cases, it is said that you’re working with a high dimensional data set. However, not all features (i.e. dimensions) may contain useful information for our task.

Humans can easily visualize 1d, 2d and 3d graphs, so it is often useful to reduce the dimensions of our data points to a lower dimension and plot it on a graph.

![Graph dimensions](/img/dimensions.png)

In our case, we'll reduce each 64 dimensional data point into 2 dimensions. Intuitively, we want to ensure that these 2 'super' features hold as much information about the original 64 features (i.e. summarise them). To achieve this, we'll be using a technique known as Principal Component Analysis (PCA).

Refer to the [scikit-learn PCA documentation](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) to import the PCA class and create a PCA model.

```python
from sklearn.decomposition import ____

# Create a regular PCA model that takes two components
pca = PCA(n_components=____)
```

Once we have created a model, we must fit our data onto the model and make it transform the data to lower its dimensions. Refer to the [fit](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA.fit) and [transform](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA.transform) functions in the PCA documentation to complete the code:

```python
# Fit the data to the model
reduced_data_pca = pca._____(_____)

# Transform the data to reduce the dimensions
reduced_data_pca = pca._____(_____)
```

We can then inspect the dimensionally reduced data and see its shape:

```python
# Print out the data
print(reduced_data_pca)

# Inspect the shape
print(reduced_data_pca.____)
```

Since the data has only 2 features, it can be plotted on a simple 2d graph where the first feature is the x coordinate and the second feature is the y coordinate. Copy-paste the following code into Python:

```python
# Since we have 10 labels, we define a color for all data points belonging to each label
colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']

# for each of the 10 labels
for i in range(len(colors)):
    # Assign the first dimension of the reduced_data_pca as the x coordinate for all data points of the i'th label
    x = reduced_data_pca[:, 0][digits.target == i]
    # same operation for the second dimension as the y coordinate
    y = reduced_data_pca[:, 1][digits.target == i]
    # plot the data points
    plt.scatter(x, y, c=colors[i])

plt.legend(digits.target_names, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title("PCA Scatter Plot")

# Show the plot
plt.show()
```

You should see something like this:

![PCA plot](/img/pca.png)

# Task 2: Unsupervised Learning (Clustering)

## Step 1: Figuring out what to do

Reducing the dimension of the data and plotting it in 2d shows us that the data points sort of group together, but there is quite some overlap. We know that there are 10 possible digits labels to assign to the data points, and (since we are studying unsupervised learning) we have no access to the labels. Obviously, our problem fits into the general class of Clustering problems, where we want to build a model to group or cluster the data points into a fixed number of classes.

To determine what algorithm to use, we can follow [this scikit-learn machine learning map](http://scikit-learn.org/stable/tutorial/machine_learning_map/).

Starting from the Clustering category:
1. We know the number of categories (since there are 10 possible digits labels)
2. We have less than 10,000 data samples

We arrive at one of the classical clustering algorithms: the K-Means algorithm.

Note that this map does require you to have some knowledge about the algorithms that are included in the scikit-learn library. This, by the way, also holds some truth for applying machine learning in general: if you have no idea what is possible, it will be very hard to decide on what your use case will be for the data.

### What is K-Means Clustering?

It is one of the simplest and widely used unsupervised learning algorithms to solve clustering problems. The procedure follows a simple and easy way to classify a given data set through a certain number of clusters that you have set before you run the algorithm. This number of clusters is called k and you select this number at random.

Then, the k-means algorithm will find the nearest cluster center for each data point and assign the data point closest to that cluster.

Once all data points have been assigned to clusters, the cluster centers will be recomputed. In other words, new cluster centers will emerge from the average of the values of the cluster data points. This process is repeated until most data points stick to the same cluster. The cluster membership should stabilize.

![K-Means GIF](/img/kmeans.gif)

However, before you can go into making a model for your data, you should definitely take a look into preparing your data for this purpose.

## Step 2: Data Preprocessing

### Data Normalization

The first thing that we’re going to do is preprocessing the data. You can standardize the digits data by, for example, making use of the scale() method:

```python
# Import the scale class
from sklearn.preprocessing import _____

# Apply `scale()` to the `digits` data
data = scale(_____)
```

By scaling the data, you shift the distribution of each feature to have a mean of zero and a standard deviation of one (unit variance). You can compare a data point before and after scaling to understand this better, but the intuition is that some features may have a wider range of values or exclusively positive or negative values. We want to normalize the distribution of all features to be similar so that any machine learning models we use will give equal consideration to each feature. 

![Scaling](/img/preprocessing.jpeg)

### Splitting into Training and Testing data

In order to assess your model’s performance later, you will also need to divide the data set into two parts: a training set and a test set. The first is used to train the system, while the second is used to evaluate the learned or trained system.

In practice, the division of your data set into a test and a training sets is disjoint: the most common splitting choice is to take 70% to 80% of your original data set as the training set, while the 30% to 20% that remains will compose the test set.

The output of the train_test_split function can be a bit confusing, so make sure you check out [the documentation](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html).

```python
# Import `train_test_split`
from sklearn.cross_validation import ______

# Split the `digits` data into training and test sets
X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(data, digits.target, digits.images, test_size=0.25, random_state=42)
```

We can quickly inspect the numbers before we move on to clustering our freshly preprocessed and split data:

```python
# Number of training features
n_samples, n_features = X_train.shape

# Print out `n_samples`
print(n_samples)

# Print out `n_features`
print(n_features)

# Number of Training labels
n_digits = len(np.unique(y_train))

# Inspect `y_train`
print(len(y_train))

# ...and any other aspects you want to investigate.
```

## Step 3: Clustering the data

After all these preparation steps, you have made sure that all your known (training) data is stored. No actual model or learning was performed up until this moment.

Now, it’s finally time to find those clusters of your training set. Use KMeans() from the cluster module to set up your model. Fill in the blanks using the [scikit-learn KMeans documentation](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html).

```python
# Import the `cluster` module
from sklearn import _____

# Create the KMeans model
clf = cluster._____(n_clusters=______, init='k-means++', random_state=42)

# Fit the training data to the model
# Hint: See documentation for the fit() function
clf.______(______)
```

What should have happened is that the K-Means model learnt 10 clusters. When shown an unknown data point, the model should be able to predict its cluster based on what it has learnt.

We can even visualize the images of the 10 cluster centers (the average of all digits in the training set assigned to that cluster):

```python
# Import matplotlib
import matplotlib.pyplot as plt

# Figure size in inches
fig = plt.figure(figsize=(8, 3))

# Add title
fig.suptitle('Cluster Center Images', fontsize=14, fontweight='bold')

# For all labels (0-9)
for i in range(10):
    # Initialize subplots in a grid of 2X5, at i+1th position
    ax = fig.add_subplot(2, 5, 1 + i)
    # Display images
    ax.imshow(clf.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)
    # Don't show the axes
    plt.axis('off')

# Show the plot
plt.show()
```

We can see something that vaguely represents various digits, but it does not seem accurate:

![Cluster Centers](/img/kmeans.png)

## Step 4: Predicting results for the test set

The next step is to predict the labels for the test set. Refer to the [KMeans.predict documentation](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans.predict) to fill in the blanks: 

```python
# Predict the labels for `X_test`
y_pred = clf.______(______)

# Print out the first 100 instances of `y_pred`
print(y_pred[:100])

# Print out the first 100 instances of `y_test`
print(y_test[:100])
```

In the code chunk above, you predict the values for the test set, which contains 450 samples. You store the result in y_pred. You also print out the first 100 instances of y_pred and y_test and you immediately see some results.

Maybe a visualization would be more helpful. We can again use PCA to reduce the dimensions of our training data to 2d, and compare the actual training labels to the ones predicted by the trained K-Means model: 

```python
# Model and fit the `digits` data to the PCA model
X_pca = PCA(n_components=2).fit_transform(X_train)

# Compute cluster centers and predict cluster index for each sample
clusters = clf.fit_predict(X_train)

# Create a plot with subplots in a grid of 1X2
fig, ax = plt.subplots(1, 2, figsize=(8, 4))

# Adjust layout
fig.suptitle('Predicted Versus Training Labels', fontsize=14, fontweight='bold')
fig.subplots_adjust(top=0.85)

# Add scatterplots to the subplots
ax[0].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters)
ax[0].set_title('Predicted Training Labels')
ax[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_train)
ax[1].set_title('Actual Training Labels')

# Show the plots
plt.show()
```

![2d plot of K-Means model](/img/kmeans-2d.png)

At first sight, the visualization doesn’t seem to indicate that the model works well.

But this needs some further investigation.

## Step 5: Evaluating the model

We shall use a confusion matrix to numerically analyze the degree of correctness of the model’s predictions.

![Toy confusion matrix](/img/confusion-matrix.png)

Each row of the matrix represents the instances in a predicted class while each column represents the instances in an actual class (or vice versa). The name stems from the fact that it makes it easy to see if the system is confusing two classes (i.e. commonly mislabelling one as another).

Refer to the [confusion_matrix documentation](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) and fill in the blanks to generate one for our predicted labels vs the true labels:

```python
# Import `metrics` from `sklearn`
from sklearn import metrics

# Print out the confusion matrix with `confusion_matrix()`
print(metrics.confusion_matrix(y_true=______, y_pred=______))
```

At first sight, the results seem to confirm our first thoughts that you gathered from the visualizations. Only the digit 5 was classified correctly in 41 cases. Also, the digit 8 was classified correctly in 11 instances. But this is not really a success.

Although there are more complicated ways to evaluate the K-Means model, it is obvious at this point that the unsupervised clustering approach to this problem does not work very well at all. We should move on to a supervised approach where we make use of that fact that we have labelled data.

# Task 3: Supervised Learning (Classification)

When you recapped all of the information that you gathered out of the data exploration, you saw that you could build a model to predict which group a digit belongs to without you knowing the labels. And indeed, you just used the training data and not the target values to build your K-Means model.

If you now decide to use the target values and follow the algorithm map along the Classification route, you’ll see that the first model that you meet is the linear SVC (which stands for Support Vector Classification).

SVCs are based on a type of supervised learning algorithm called a Support Vector Machine (SVM). Intuitively, an SVM and most other classification algorithms will try to find a boundary that best separates the various classes present in our data.

![SVM](/img/svm.gif)

We shall now be implementing a linear SVC model to perform digit classification. If you are feeling comfortable with the scikit-learn syntax and documentation already, you may want to try out some other classification algorithm such as Logistic Regression or K Nearest Neighbours.

## Step 1: Building and training the model

Import the svm class from scikit-learn and create an SVC model object using the [svm.SVC documentation](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html):

```python
# Import the `svm` model
from sklearn import ______

# Create the SVC model with a `linear` kernel
svc_model = svm._______(kernel=______, gamma=0.001, C=100.)
```

Next, use the fit() method to start training the svc_model on the training samples (X_train) and the associated labels (y_train).

Fill in the blanks based on the [SVC.fit documentation](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC.fit)

```python
# Fit the data to the SVC model
svc_model.______(______, _____)
```

We have to make use of X_train and y_train to fit the data to the SVC model. This is clearly different from clustering, where we did not use the training labels y_train at all. 

Note also that in this example, you set the value of gamma and C manually. These are mathematical parameters for the SVM model, which can be optimally selected to obtain the best performing model.

## Step 2: Predict results for test data

The next step is to predict the labels for the test set. Refer to the [SVC.predict documentation](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC.predict) to fill in the blanks: 

```python
# Predict the label of `X_test`
print(svc_model.______(______))

# Print `y_test` to check the results
print(y_test)
```

The SVC model also has a method called score() to quickly give us the mean accuracy over the test data. Use the [SVC.score documentation](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC.score) to obtain accuracy:

```python
# Apply the classifier to the test data, and view the accuracy score
svc_model.score(______, ______)
```

### Messing with model parameters

A kernel is a similarity function, which is used to compute similarity between the training data points. When you provide a kernel to an algorithm, together with the training data and the labels, you will get a classifier, as is the case here. For the SVM, you will typically try to linearly divide your data points. However, some other types of kernels can be polynomial or based on radial basis functions. The C and gamma parameters are associated with the choice of kernel.

Experimenting with the values of these parameters can lead to better/worse results. Try it yourself!

```python
# Train and score a new classifier with some different parameters
svm.SVC(C=10, kernel='rbf', gamma=0.001).fit(X_train, y_train).score(X_test, y_test)
```

## Step 4: Evaluating the model

We already saw that the SVC model has a very high mean accuracy for the test data. We can dive deeper and see how it performs for each of the 10 digits labels by printing a confusion matrix exactly as before: 

```python
# Assign the predicted values to `predicted`
predicted = svc_model.predict(X_test)

# Print the confusion matrix
print(metrics.confusion_matrix(______, ______))
```

## Step 5: Visualize the results

Like for K-Means, we can again visualize the predicted vs actual labels to get a rough idea of how our model performs.

```python
# Import `Isomap()`
from sklearn.manifold import Isomap

# Create an isomap and fit the `digits` data to it
X_iso = Isomap(n_neighbors=10).fit_transform(X_train)

# Compute cluster centers and predict cluster index for each sample
predicted = svc_model.predict(X_train)

# Create a plot with subplots in a grid of 1X2
fig, ax = plt.subplots(1, 2, figsize=(8, 4))

# Adjust the layout
fig.subplots_adjust(top=0.85)

# Add scatterplots to the subplots
ax[0].scatter(X_iso[:, 0], X_iso[:, 1], c=predicted)
ax[0].set_title('Predicted labels')
ax[1].scatter(X_iso[:, 0], X_iso[:, 1], c=y_train)
ax[1].set_title('Actual Labels')

# Add title
fig.suptitle('Predicted versus actual labels', fontsize=14, fontweight='bold')

# Show the plot
plt.show()
```

We can see that predicted labels coincide pretty well with the actual labels. 

![SVM 2D Plot](/img/svm-plot.png)

We can also see specific data points and their corresponding predictions from the test set:

```python
# Assign the predicted values to `predicted`
predicted = svc_model.predict(X_test)

# Zip together the `images_test` and `predicted` values in `images_and_predictions`
images_and_predictions = list(zip(images_test, predicted))

# For the first 4 elements in `images_and_predictions`
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    # Initialize subplots in a grid of 1 by 4 at positions i+1
    plt.subplot(1, 4, index + 1)
    # Don't show axes
    plt.axis('off')
    # Display images in all subplots in the grid
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    # Add a title to the plot
    plt.title('Predicted: ' + str(prediction))

# Show the plot
plt.show()
```

![Predicted digits](/img/svm-preds.png)

# Next Steps

Congratulations! You just build your first machine learning application to recognize handwritten digits. It may not seem like much, but this algorithm actually *learnt* the visual difference between '6' and '8', for example. I think thats really amazing!

In general, machine learning is being applied to so many interesing and important problems. It informs everything from our Facebook feed, our suggested traffic routes in Google Maps, our autopilot email spam filters, and even the security of our banking information. Here's a [quick rundown of some cool applications](https://chatbotnewsdaily.com/how-artificial-intelligence-ai-machine-learning-ml-making-our-daily-life-better-773fe5bc3bcd) in our daily lives.

Also check out this brilliant visual recap of some of the basic concepts of machine learning: http://www.r2d3.us/visual-intro-to-machine-learning-part-1/

## Reading more about machine learning

The internet is your friend! There are so many fantastic stand-alone articles about understanding concepts, implementing algorithms and all other aspects of machine learing. If somebody has taken the time to write a Medium article or a blog post, it usually has good content. 

A quick google search for 'introduction to machine learning python' will get you started!

I personally used the [scikit-learn website](http://scikit-learn.org/stable/documentation.html) to familiarize myself with the various models and concepts. It has many examples and tutorials on how to harness the full power of the library. There is atleast one example of each machine learning technique and the most important functions for preprocessing, splitting and evaluating.

Another fantastic website to start your journey in machine learning is [Kaggle](https://www.kaggle.com/).

## Finding datasets

The [UCI Machine Learning repository](http://archive.ics.uci.edu/ml/datasets.html) has lots of interesting datasets you can immediately download and start to play around with. Kaggle also has many free datasets available.

## Studying the mathematics behind the models

For anything beyond simple applications, its true that you need a deeper understanding of the mathematics and concepts behind machine learning. The Machine Learning (CZ4041) module at NTU is a pretty good introduction, and the following free online courses are also very highly recommended:
1. Andrew Ng's legendary [Coursera course](https://www.coursera.org/learn/machine-learning)
2. Sebastian Thrun's legendary [Udacity course](https://www.udacity.com/course/intro-to-machine-learning--ud120)	
