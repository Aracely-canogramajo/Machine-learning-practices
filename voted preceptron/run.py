import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def run(Xtrain_file, Ytrain_file, test_data_file, test_label, pred_file):  
  k = 0;
  t = 0;
  w = [];
  c = [];  
  w.append(np.zeros(Xtrain_file[0].shape));
  c.append(0); 
  T = 1; # choose  epoch T   

  #changing labels from 0/1 to -1/1
  q = 0;
  for line in Ytrain_file:
    if Ytrain_file[q] == 0:
      Ytrain_file[q] = -1;
    q += 1;
  
  #training
  while t <= T:
    i = 0;
    for line in Xtrain_file:
      dot = np.dot(w[k], Xtrain_file[i]);
      y_pred = np.sign(dot);
      if(y_pred == Ytrain_file[i]):
        c[k] = c[k] + 1;
      else:
        w.append(w[k] + Ytrain_file[i] * Xtrain_file[i]);
        c.append(1);
        k = k + 1;
      i = i + 1;
    t = t + 1; 
       
  #prediction
  predict = np.empty(len(test_data_file));
  index = 0;
  for y in test_data_file:
    sum = 0;
    for i in range(k):
      sum += c[i] * np.sign(np.dot(w[i], test_data_file[index]));
    pred = np.sign(sum); 
    if pred == -1:
      pred = 0;
    predict[index] = int(pred);
    index += 1;  
  
  np.savetxt(pred_file, predict, fmt='%1d', delimiter=","); #output test results
  
  # confusion matrix and accuracy
  print(confusion_matrix(test_label, predict, labels=[-1, 1]));
  print(accuracy_score(test_label, predict));
  
def main(Xtrain_file, Ytrain_file, pred_file):
  # Reading data=
  xtrain = np.genfromtxt(Xtrain_file, dtype=int, delimiter=",");
  ytrain = np.genfromtxt(Ytrain_file, dtype=int, delimiter=",");
  
  training_size = int((xtrain.size/xtrain[0].size)/2);
  
  # split training data into training data and testing data
  train = np.empty(training_size); # create array for training data
  train_label = np.empty(training_size); # create array for training label
  test = np.empty(training_size); # create array for test data
  test_label = np.empty(training_size); # create array for test label
  
  temp = np.array_split(xtrain, 10); #split xtrain into two
  temp2 = np.array_split(temp[0], 10);
  
  train = temp[0]; 
  train = np.append(train, temp[1], axis = 0); # 2%
  train = np.append(train, temp2[2], axis = 0); # 2%
  train = np.append(train, temp2[3], axis = 0); # 2%
  train = np.append(train, temp2[4], axis = 0); # 2%
  train = np.append(train, temp2[5], axis = 0); # 2%
  train = np.append(train, temp2[6], axis = 0); # 2%
  train = np.append(train, temp2[7], axis = 0); # 2%
  train = np.append(train, temp2[8], axis = 0); # 2%
  
  test = temp[9]; #save last 10% into test
  
  
  temp = np.array_split(ytrain,10); #split the labels into two
  temp2 = np.array_split(temp[0], 10);
  
  train_label = temp[0];
  train_label = np.append(train_label, temp[1], axis = 0); # 2%
  train_label = np.append(train_label, temp2[2], axis = 0); # 2%
  train_label = np.append(train_label, temp2[3], axis = 0); # 2%
  train_label = np.append(train_label, temp2[4], axis = 0); # 2%
  train_label = np.append(train_label, temp2[5], axis = 0); # 2%
  train_label = np.append(train_label, temp2[6], axis = 0); # 2%
  train_label = np.append(train_label, temp2[7], axis = 0); # 2%
  train_label = np.append(train_label, temp2[8], axis = 0); # 2%
  
  
  test_label = temp[9];
    
  # call run
  run(train, train_label, test, test_label, pred_file);
  
  


if __name__ == "__main__":
  
  Xtrain_file = "Xtrain.csv"
  Ytrain_file = "Ytrain.csv"  
  pred_file = 'result'
  pred_file = 'result'
  
  main(Xtrain_file, Ytrain_file, pred_file)
  # run(Xtrain_file, Ytrain_file, test_data_file, pred_file)