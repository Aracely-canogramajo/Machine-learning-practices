import numpy as np
from sklearn.metrics import confusion_matrix

# This will be the code you will start with, note that the formate of the function 
# run (train_dir,test_dir,pred_file) and saving method can not be changed
# Feel free to change anything in the code

def run (train_input_dir,train_label_dir,test_input_dir, test_label_dir, pred_file):
    # Reading data
    test_data = np.loadtxt(test_input_dir)
    test_label = np.loadtxt(test_label_dir)
    train_input = np.loadtxt(train_input_dir)
    train_label = np.loadtxt(train_label_dir)

    [num] = test_data[0].shape
    print(num)
    #print(test_data)
    
    # finding the centroid -------------------------------------------------------
    [t] = train_input[0].shape #get num of features
    print(t)
    
    arr_class0= np.empty(t) #to put the values in respective classes
    arr_class1= np.empty(t) #initialize to size of features
    arr_class2= np.empty(t) 
    total0 = 0
    total1 = 0
    total2 = 0
    
    x = 0
    for line in train_input:
      #print(line)
      #print(train_label[x])
      
      if(train_label[x] == 0):
        total0+=1;
        i = 0;
        for temp in line:
          arr_class0[i] += temp;
          i+=1; 
      
      if(train_label[x] == 1):
        total1+=1;
        i = 0;
        for temp in line:
          arr_class1[i] += temp;
          i+=1;
          
      if(train_label[x] == 2):
        total2+=1;
        i = 0;
        for temp in line:
          arr_class2[i] += temp;
          i+=1;
      
      x += 1;
      
    #print(arr_class0)
    #print(arr_class1)
    #print(arr_class2)
    #print(total0)
    #print(total1)
    #print(total2)
    
    #calculating centroid
    temp = 0;
    for j in arr_class0:
      arr_class0[temp] = arr_class0[temp] / total0;
      temp+=1;
    
    temp = 0;
    for j in arr_class1:
      arr_class1[temp] = arr_class1[temp] / total1;
      temp+=1;
    
    temp = 0;
    for j in arr_class2:
      arr_class2[temp] = arr_class2[temp] / total2;
      temp+=1;
      
    print(arr_class0); #centroid 0
    print(arr_class1); #centroid 1
    print(arr_class2); #centroid 2
    
   
    
    # Calculating nearest neighbor --------------------------------------------
    p = np.empty(len(test_data));
    k = 0;
    
    for line in test_data:
      #print(line)
      #print(arr_class0)
      d0 = np.empty(t);
      d1 = np.empty(t);
      d2 = np.empty(t);
      d0 = np.linalg.norm(line - arr_class0);
      d1 = np.linalg.norm(line - arr_class1);
      d2 = np.linalg.norm(line - arr_class2);
      
      #print(d0);
      #print(d1);
      #print(d2);
      
      if(d0 <= d1):
        if(d0 <= d2):
          # print("0");
          p[k] = 0;
        else:
          # print("2");
          p[k] = 2;
      else:
        if(d1 <= d2):
          # print("1");
          p[k] = 1;
        else:
          # print("2");
          p[k] = 2;
      
      k+=1;
      
    prediction = p;
    print(p);
    
    # Saving you prediction to pred_file directory (Saving can't be changed)
    np.savetxt(pred_file, prediction, fmt='%1d', delimiter=",");
    
    # label = np.array([2,1]);
    # print(confusion_matrix(label, p, labels=[0,1,2]));
    print(confusion_matrix(test_label, p, labels=[0,1,2]));

    
if __name__ == "__main__":
    train_input_dir = 'training1.txt'
    train_label_dir = 'training1_label.txt'
    test_input_dir = 'testing1.txt'
    test_label_dir = 'testing1_label.txt'
    pred_file = 'result'
    run(train_input_dir,train_label_dir,test_input_dir, test_label_dir,pred_file)
