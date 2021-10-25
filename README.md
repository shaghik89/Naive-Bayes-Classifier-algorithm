# Naive-Bayes-Classifier-Algorithm

# Introducing Bayes Theorem:
- Bayes' Theorem gives the conditional probability of an event A given another event B has occured. 
- Conditional Probabilty is the likelihood of an outcome occuring, with respect to previouse outcome occuring.
- Bayes' theorem allows you to update predicted probabilities of an event by incorporating new information.

In mathematical notation, the probability of an event, :math:`A`, is denoted by
:math:`P(A)`

The theorem is:

.. math::

    posterior = \frac{likelihood \times prior}{normalization}

or:

.. math::

    P(A|B) = \frac{P(B|A)P(A)}{P(B)}

To compute the right hand side of the equation you'll need to estimate the
prior, the likelihood, and the normalization.
