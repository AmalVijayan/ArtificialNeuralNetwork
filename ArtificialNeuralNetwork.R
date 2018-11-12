
################################ ANN #############################################


########################## Data Preprocessing ################################# 

#1 importing the datset
dataset = read.csv('Churn_Modelling.csv')


#2 Removing all unnecessary variables
dataset = dataset[,4:14]


#3 Dealing with missing and categirical data

dataset$Geography = as.numeric(factor(dataset$Geography,
                                      levels = c('France','Germany','Spain'),
                                      labels = c(1,2,3)))

dataset$Gender = as.numeric(factor(dataset$Gender,
                                      levels = c('Female','Male'),
                                      labels = c(1,2)))


#4 Splitting into training and test sets
library(caTools)
set.seed(1234)
split = sample.split(dataset$Exited, SplitRatio = 0.8)
training_set = subset(dataset,split == TRUE)
test_set = subset(dataset,split == FALSE)


#5 Feature scaling

training_set[-11] = scale(training_set[-11])
test_set[-11] = scale(test_set[-11])




####################################### Artificial Neural Network ############################################

#importing the libraries
#install.packages('h2o')
library(h2o)

#2 initializing the ANN Classifier
h2o.init(nthreads = -1)

#3 Create a classifier for ANN
classifier = h2o.deeplearning(y = 'Exited', training_frame = as.h2o(training_set), 
                              activation = 'Rectifier', 
                              hidden = c(6,6),
                              train_samples_per_iteration = 10,
                              epochs = 20)

#4 predicting the result for test set

prob = h2o.predict(classifier, as.h2o(test_set))
Y_pred = as.vector((prob > 0.5))

#5 confusion matrix
cm = table(test_set[,11], Y_pred)