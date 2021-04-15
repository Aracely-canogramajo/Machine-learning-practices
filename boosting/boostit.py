import numpy as np
from sklearn.metrics import confusion_matrix

class BoostingClassifier:
  """ Boosting for binary classification.
  Please build an boosting model by yourself.

  Examples:
  The following example shows how your boosting classifier will be used for evaluation.
  >>> X_train, y_train = load_train_dataset() # we ignore the process of loading datset
  >>> X_test, y_test = load_test_dataset()
  >>> clf = BoostingClassifier().fit(X_train, y_train)
  >>> y_pred =  clf.predict(X_test) # this is how you get your predictions
  >>> evaluation_score(y_pred, y_test) # this is how we get your final score for the problem.

  """
  def __init__(self):
    # initialize the parameters here
    self.T = 10; # choose  epoch T
    self.w = [];
    self.a = [];
    self.M = [];

  
  def fit(self, X, y):
    """ Fit the boosting model.

    Parameters
    ----------
    X : { numpy.ndarray } of shape (n_samples, n_features)
        The input samples with dtype=np.float32.
    
    y : { numpy.ndarray } of shape (n_samples,)
        Target values. By default, the labels will be in {-1, +1}.

    Returns
    -------
    self : object
    """
    size = X.size/X[0].size;
    weight = 1/abs(size); # inital weights
    temp = np.zeros(int(size)); 
    index = 0;
    for ara in temp:
      temp[index] = weight;
      index+=1;
    self.w.append(temp); # the whole array has same weight size
    weighted_error = [];
    T_VP = 2; # choose  epoch T 
    t = 0;
    
    while t <= self.T:
      print("Iteration ", t, ":");
      #run learning algorithm(KNN learning algorithm) --------------------------------------
      k = 0;
      t_VP = 0;
      w = [];
      c = [];  
      w.append(np.zeros(X[0].shape));
      c.append(0); 
        
      #training
      while t_VP <= T_VP:
        i = 0;
        for line in X:
          dot = np.dot(w[k], X[i]);
          y_pred = np.sign(dot);
          if(y_pred == y[i]):
            c[k] = c[k] + 1;
          else:
            w.append(w[k] + y[i] * X[i]);
            c.append(1);
            k = k + 1;
          i = i + 1;
        t_VP = t_VP + 1; 
           
      #prediction
      p = np.empty(len(X));
      index = 0;
      for j in X:
        sum = 0;
        for i in range(k):
          sum += c[i] * np.sign(np.dot(w[i], X[index]));
        pred = np.sign(sum); 
        p[index] = int(pred);
        index += 1;  
      
      self.M.append(p);
      
      # calculate weighted error ------------------------------------------------
      # print(y);
      # print(p);
      b = p == y;
      # print(b);
      
      correct = (p==y).sum();
      errors = y.size - correct;
      error_rate = errors / y.size;
      
      print("Error = ", error_rate);
      
      sum_of_error_weights = 0;
      total_sum_of_weights = 0;      
      index = 0;
      for line in self.w[t]:    
        if b[index] == False:
          sum_of_error_weights += line;
          total_sum_of_weights += line;
          index += 1;
          continue;
        total_sum_of_weights += line;
        index += 1;
        
      # print("sum_of_error_weights ",sum_of_error_weights);
      # print("total_sum_of_weights ", total_sum_of_weights);
      # print("weighted_error ",weighted_error);
      # print("div ",sum_of_error_weights/total_sum_of_weights);
      weighted_error.append(sum_of_error_weights/total_sum_of_weights);
      # print(weighted_error);    
      
      # check weighted error ----------------------------------------------------
      # if weighted_error[t] >= 1/2:
        # print("weighted_error >= 1/2");
        # self.T = t -1;
        # break;
      
      # find confidence & increase weights --------------------------------------
      confidence = 0.5*np.log((1-weighted_error[t])/weighted_error[t]);
      print("Alpha = ", confidence);
      self.a.append(confidence);
           
      temp_arr = np.zeros(int(size));      
      index = 0;
      i = 0;
      for line in self.w[t]:
        new_missclassified = 0;
        new_correctly_class = 0;
        if b[i] == True:
          new_correctly_class = line/(2*(1-weighted_error[t]));
          temp_arr[index] = new_correctly_class;
        if b[i] == False:
          new_missclassified = line/(2*weighted_error[t]);
          temp_arr[index] = new_missclassified;
        i += 1;
        index += 1;
        
      print("Factor to increase weights = ", new_missclassified);
      print("Factor to decrease weights = ", new_correctly_class);

      self.w.append(temp_arr);
      
      # print(self.w);
      
      T_VP += 2;
      t += 1;

    return self

  def predict(self, X, y):
    """ Predict binary class for X.

    Parameters
    ----------
    X : { numpy.ndarray } of shape (n_samples, n_features)

    Returns
    -------
    y_pred : { numpy.ndarray } of shape (n_samples)
             In this sample submission file, we generate all ones predictions.
    """
    print("Testing");    
    sum = 0;
    index = 0;
    for i in self.M:
      sum += self.a[index] * self.M[index];

    c = confusion_matrix(np.sign(sum),y);
    fp = c[1][0]; 
    fn = c[0][1];
    p = c[0][0] + c[1][0];
    n = c[0][1] + c[1][1];
    
    print(c);
    print("False Positives: ", fp);
    print("False Negatives: ", fn);
    print("Error rate: ", (fp+fn)/(n+p), "%");
    
    return(np.sign(sum));
    
    # return np.ones(X.shape[0], dtype=int)


