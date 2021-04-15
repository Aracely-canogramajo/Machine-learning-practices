import numpy as np
from sklearn.metrics import confusion_matrix

def run(Xtrain_file, Ytrain_file, test_data_file, test_label, pred_file):
  # choose K
  k = 4;
  
  p = np.empty(len(test_data_file)); #make prediction array
  index = 0;  
  line = 0;
  for h in test_data_file:
    a = np.zeros(shape=(Ytrain_file.size,2)) # create array to store distance & class    
    temp = 0;
    for x in Xtrain_file:
      dist = np.linalg.norm(h - x); #find euclidean distance
      a[temp] = [dist, Ytrain_file[temp]]; # save each dist & class into array
      temp += 1;
      
    
    a = a[np.argsort(a[:,0])] # sort array by distance
    classes = np.zeros(11); #make array for classes
    
    for i in range(k):
      j = int(a[i][1]);
      classes[j] += 1;
      
    max_class = np.argmax(classes);
    p[index] = max_class;
    index += 1;
    line += 1;
    
    
  # output test results 
  np.savetxt(pred_file, p, fmt='%1d', delimiter=","); 

  print(confusion_matrix(test_label, p, labels=[1,2,3,4]));


def main(Xtrain_file, Ytrain_file, pred_file):
  print("in main");
  xtrain = np.genfromtxt(Xtrain_file, dtype= float, delimiter=",");
  ytrain = np.genfromtxt(Ytrain_file, dtype= float, delimiter=",");

  training_size = int((xtrain.size/xtrain[0].size)/2);
  
  print("training size ", training_size);
  
  # shape=(Ytrain_file.size,2
  
  # split training data into training data and testing data
  train = np.empty(training_size); # create array for training data
  train_label = np.empty(training_size); # create array for training label
  test = np.empty(training_size); # create array for test data
  test_label = np.empty(training_size); # create array for test label
  
  temp = np.array_split(xtrain, 8); #split xtrain into two
  train = temp[0]; # save half into train
  train = np.append(train, temp[2], axis = 0);
  train = np.append(train, temp[4], axis = 0);
  train = np.append(train, temp[6], axis = 0);
  test = temp[1]; #save other half into test
  test = np.append(test,temp[3], axis = 0);
  test = np.append(test,temp[5], axis = 0);
  test = np.append(test,temp[7], axis = 0);
  
  
  
  temp = np.array_split(ytrain,8); #split the labels into two
  train_label = temp[0];
  train_label = np.append(train_label, temp[2], axis = 0);
  train_label = np.append(train_label, temp[4], axis = 0);
  train_label = np.append(train_label, temp[6], axis = 0);
  test_label = temp[1];
  test_label = np.append(test_label, temp[3], axis = 0);
  test_label = np.append(test_label, temp[5], axis = 0);
  test_label = np.append(test_label, temp[7], axis = 0);
  
  # for line in train_label:
    # print (line)
  


  # call run
  run(train, train_label, test, test_label, pred_file);
  

if __name__ == "__main__":
  
  Xtrain_file = "Xtrain.csv"
  Ytrain_file = "Ytrain.csv"  
  pred_file = 'result'
  
  main(Xtrain_file, Ytrain_file, pred_file)
  # run(Xtrain_file, Ytrain_file, test_data_file, pred_file)