#
# kNN
#
# Matthew Sudmann-Day
#
# Performs k-Nearest Neighbor predictions based on training data and test features.
#
# Function kNN()
#
# Parameters:
#   features: the two-dimensional features from the training set
#   labels: the correctly classified labels from the training set
#   k: the number of nearest neighbors from which to select a majority prediction
#   p: the 'power' used to define the distance method
#       1=Manhattan distance
#       2=Euclidian distance
#       Inf=Chebyshev distance
#       any other positive number=similar to Euclidian but using power other than 2.
#   predFeatures: the features of the prediction set for which we need to provide
#                 labels
#
# Returns a list containing:
#   predLabels: a vector of predicted labels correlating with predFeatures
#   prob: a vector of probabilities corresponding with predLabels
#
# Uses R packages:
#   assertthat (for assertions on parameters)
#   eeptools (for the statamode() function which provides majority votes from a vector)
#
# The libraries are loaded outside the function.
library(assertthat)
library(eeptools)
kNN <- function(features, labels, k, p, predFeatures)
{
  # Reference the required libraries.
  library(assertthat, quietly=TRUE)
  library(eeptools, quietly=TRUE)
  
  nPredFeatures <- nrow(predFeatures)
  
  # Assert that the training features, training labels, and prediction features
  # are consistent with each other and have the necessary dimensionality.
  not_empty(features)
  not_empty(labels)
  not_empty(predFeatures)
  assert_that(ncol(features) == 2)
  assert_that(ncol(predFeatures) == 2)
  assert_that(nrow(features) == length(labels))
  
  # Assert that k is a positive integer and that p > 0.
  is.count(k);
  assert_that(p > 0)
  
  # Create a distances matrix, initially containing all NA values.
  dist <- matrix(NA, nPredFeatures, nrow(features))
  
  # Fill the matrix in different ways based on the value of p.
  #
  # The case of p==1 works the same as all p<Inf, but it requires less computation
  # when handled as the special case.
  if (p == 1)
  {
    # Loop through all test features.  The loop is inside the conditional on p for performance.
    for (i in 1:nPredFeatures)
    {
      # Store the Manhattan distances for each training feature to the test feature.
      dist[i,] <- abs(features[,1]-predFeatures[i,1])
                  + abs(features[,2]-predFeatures[i,2])
    }
  }
  else if (p == Inf)
  {
    # Loop through all test features.  The loop is inside the conditional on p for performance.
    for (i in 1:nPredFeatures)
    {
      # Store the Chebyshev distances for each training feature to the test feature.
      dist[i,] <- pmax.int(abs(features[,1]-predFeatures[i,1]),
                           abs(features[,2]-predFeatures[i,2]))
    }
  }
  else
  {
    # Precalculate the reciprocal of p to reduce computation inside the loop below.
    p_recip <- 1/p

    # Loop through all test features.  The loop is inside the conditional on p for performance.
    for (i in 1:nPredFeatures)
    {
      # Store the Euclidean distances (or other-powered distances if p!=2) for each training
      # feature to the test feature.
      dist[i,] <- (abs(features[,1]-predFeatures[i,1])^p
                   + abs(features[,2]-predFeatures[i,2])^p) ^ p_recip
    }
  }

  # The k==1 case is handled separately because it is much simpler and faster than the
  # general case, regardless of the number of categories.
  if (k == 1)
  {
    # Find the index of the minimum distance for each set of training features.
    kNeighbors <- apply(dist, 1, which.min)
    
    # Look up the labels corresponding to that indices.
    kLabels <- labels[kNeighbors]
    
    # Return those labels, along with a probability of 1 for every label.
    return(list(predLabels=kLabels, prob=rep(1, length(kLabels))))
  }
  
  # Sort the distances in the distance matrix by set of test features (columns), extract
  # the indices of those distances, and then throw away all but the first k of each.
  kNeighbors <- apply(dist, 1, order)[1:k,]

  # Create a matrix of labels with appropriate dimension.
  kLabels <- matrix(NA, k, nPredFeatures)
  kLabels[,] <- labels[kNeighbors]

  # Binary classification is handled separately (but only when the labels are 0 and 1)
  # because simple arithmetic applies that cannot be used in the general case.
  if (min(labels) == 0 && max(labels) == 1)
  {
    avg <- apply(kLabels, 2, mean)
    predLabels <- round(avg)
    prob <- abs(1 - avg - predLabels)
  }
  else
  {
    # Reverse the labels so that the nearest is last in the list in order to use a feature
    # of the statamode() function.
    kLabels <- apply(kLabels, 2, rev)
    # Apply the statamode function to each group of labels to find the mode, or majority,
    # classification.  When there is no majority, use the last in the list which has been
    # set up as the nearest neighbor.  Statamode() does not have a "first" option.
    predLabels <- as.numeric(apply(kLabels, 2, statamode, "last"))
    # Determine which of the k neighbors that match the chosen label...
    matches <- apply(kLabels, 1, `==`, as.matrix(predLabels))
    # ...and then count them.
    counts <- apply(matches, 1, sum)
    # Use the counts to determine our probability scores.
    prob <- counts/k
  }
  
  # Return predicted labels and corresponding probabilities in a list.
  return(list(predLabels=predLabels, prob=prob))
}
