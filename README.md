# Logistic-Regression
This program imports numpy and sklearn, which provides convenient methods.
I write three functions: sigmoid, costfunction,gradAscent to show the detail code of logistic-regression model.
But I choose to use sklearn.linear_model to train logistic-regression model. In linear_model.LogisticRegression, there are some parameters to set: 
- C:regularization factor 
- multi-class='ovr'or'multinomial'
- solver='lbfgs'(for both multinomial and ovr) or'liblinear' (for ovr only)

And I prepare two datasets:Iris or Wine, you can change it in function split(ratio),which can split the dataset into train set and test set according to a certain proportion. 

Then use the preprocessing function to standardscale the train set and test set before we get to train the logistic-regression model.

What's more, plot(X,Y) function can visualize the decision result in two dimensions(the first two features).
