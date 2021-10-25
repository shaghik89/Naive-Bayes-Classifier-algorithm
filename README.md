# Naive-Bayes-Classifier-Algorithm

- Naive Bayes Classifier is based on Bayes' Theorem which gives the conditional probability of an event A given B
- NB is a supervised learning algorithm and it is used in classification algorithm like weather prediction, face recognition, news prediction, cancer or heart disease. 

# Introducing Bayes Theorem:
- Bayes' Theorem gives the conditional probability of an event A given another event B has occured. 
- So is it basicaqlly calculates the conditional probability of the occurance of an event based on the prior knowledge of conditions that might be related to the event.
- Conditional Probabilty is the likelihood of an outcome occuring, with respect to previouse outcome occuring.
- Bayes' theorem allows you to update predicted probabilities of an event by incorporating new information.

In mathematical notation, the probability of an event, :math:`A`, is denoted by
:math:`P(A)`

The theorem is:

.. math::

    P(A|B) = \frac{P(B|A)P(A)}{P(B)}

To compute the right hand side of the equation you'll need to estimate the
prior, the likelihood, and the normalization.
