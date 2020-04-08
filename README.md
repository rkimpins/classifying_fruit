# Introduction
For this project, I am classifying fruit based on their measurements.
todo add more here
A short introduction, describing the background of the task

# Formulation
Problem formulation (what is input, what is output, where did you get the dataset, number of samples, etc.)
The input to this problem are measurements of different fruits. These include the mass, width, height, and color_score. The color_score is a normalized color between 0 and 1. The ouput is an integer fruit label corresponding to a fruit. For this dataset, 1-apple, 2-mandrin, 3-orange, 4-lemon. I downloadedthis dataset from https://github.com/susanli2016/Machine-Learning-with-Python, but my understanding is it was originally created by Dr.Iain Murray at the University of Edinburgh. https://www.ed.ac.uk/profile/iain-murray. An interesting challenge that I was excited to tackle with this dataset is its small size. There are only 59 samples included in this dataset, which made decisions about training set, validation set, and test set size critical. I will discuss the implications of this later on.

# Approaches
For this project, I took an exploratory approach. I tried machine learning algorithms in an attempt to find the one most appropriate to this task.

## Baseline
Approaches and baselines (what are the hyperparameters of each approach/baseline, how do you tune them)?
Lets start of with a baseline for this dataset. A quick barplot shows us that (fruit_distribution.png)the most common label is apple or lemon, with 19 outof the 59 labels. Therefore, we could achieve 32% accuracy with majority guess on the training set. If we assume that our classes will be evenly distributed in our test scenario, we can assume we achieve a 1/4 = 25% accuracy with majority guess on test cases.
## Algorithms
My approach for each of these machine learning algorithms is very similair. I randomly select a test set of size 10, and set it aside. Then, I do an exhaustive search over some hand-picked values for my hyperparameters. For each value of hyperparameter, we report its accuracy on our validation set by averaging its performance over K=10 fold cross validation. Afterwards, we return the best performing hyperparameters and run them on our test set. An important thing to note is that since our dataset is so smal, there is a huge amount of variability in our final result. In an attempt to get a more consistent picture of how each algorithm performs, I repeat the process on a randomly selected test set many times and average the accuracy on the final training test. To be clear, no additional hyperparameter tuning is being performed for this iterations, so we are not corrupting our training set with our test set. We are simply reducing variability in result to get a better idea of what the average performance would be.
## Logistic Regression
### Hyperparameters
* C: ...
* Max Iterations: Didn't make much of a difference, just set this to a level that it converged at.
### Representative Sample
0.8367346938775511 0.8
Best C: 0.8
Best validation performance: 0.7366666666666666
Test performance: 0.8
log_reg_tuning
### Averaged Performance

## SVM maxmargin classification
### Hyperparameters
* C:
* gamma:
### Representative Sample
0.8979591836734694 0.6
Best gamma: 0.2
Best C: 0.95
Best validation performance: 0.51
Test performance: 0.6

svc_tuning_gamma.png
svc_tuning_C.png
### Averaged Performance

## Decision Tree
### Hyperparameters
### Representative Sample
1.0 0.9
Best max depth: 10
Best validation performance: 0.93
Test performance: 0.9

tree_tuning.png
decision_tree.png
### Averaged Performance

## Knearest neighbours Consider implementing this
### Hyperparameters
### Representative Sample
### Averaged Performance

## Discussion (for each?)

## Conclusion

Evaluation metric (what is the measure of success, is it the real goal of the task, or an approximation? If it’s an approximation, why is it a reasonable approximation?)
Results. (What is the result of the approaches? How is it compared with baselines? How do you interpret the results?)
