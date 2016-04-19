#
# kNN_MNIST
#
# Matthew Sudmann-Day
#
# Performs k-Nearest Neighbor predictions based on training data and test features.
# This uses an adaptation of my kNN function in kNN.R.  It is adapted to support
# cross-validation and decides majority vote by inverse distance weighting.
#
# Function kNN_dist() generates a distance matrix for use in kNN.
#
# Parameters:
#   features: the two-dimensional features from the training set
#   p: the 'power' used to define the distance method
#       1=Manhattan distance
#       2=Euclidian distance
#       Inf=Chebyshev distance
#       any other positive number=similar to Euclidian but using power other than 2.
#   predFeatures: the features of the prediction set for which we need to provide
#                 labels (must be NULL if cv==TRUE)
#   cv: TRUE if this function is being used for cross-validation (FALSE by default)
#
# Returns: a distance matrix
#
# Uses R packages:
#   assertthat (for assertions on parameters)
#
kNN_dist <- function(features, p, predFeatures=NULL, cv=FALSE)
{
  # Reference the required libraries.
  library(assertthat, quietly=TRUE)
  
  if (cv)
  {
    assert_that(is.null(predFeatures))
    predFeatures = features
  }
  not_empty(features)
  assert_that(p > 0)
  
  N <- ncol(features)
  
  # Create a distances matrix, initially containing all NA values.
  dist <- matrix(NA, nrow(predFeatures), nrow(features))

  # Fill the matrix in different ways based on the value of p.
  if (p == Inf)
  {
    # Loop through all test features.  The loop is inside the conditional on p for performance.
    for (i in 1:nrow(predFeatures))
    {
      # Store the Chebyshev distances for each training feature to the test feature.
      dist[i,] <- 0
      for (j in 1:N)
      {
        dist[i,] <- pmax.int(dist[i,], abs(features[,j]-predFeatures[i,j]))
      }
    }
  }
  else
  {
    # Precalculate the reciprocal of p to reduce computation inside the loop below.
    p_recip <- 1/p
    
    # Loop through all test features.  The loop is inside the conditional on p for performance.
    for (i in 1:nrow(predFeatures))
    {
      # Store the Euclidean distances (or other-powered distances if p!=2) for each training
      # feature to the test feature.
      dist[i,] <- 0
      for (j in 1:N)
      {
        dist[i,] <- dist[i,] + abs(features[,j]-predFeatures[i,j])^p
      }
      dist[i,] <- dist[i,] ^ p_recip
    }
  }

  return(dist)
}

# Function kNN_eval() makes kNN predictions based on a provided distance matrix.
# From the k nearest neighbors, it applies an inverse distance weighting
# formula.
#
# Parameters:
#   dist: the distance matrix return by kNN_dist()
#   features: the two-dimensional features from the training set
#   labels: the correctly classified labels from the training set
#   uniqueLabels: a unique list of all possible labels
#   k: the number of nearest neighbors from which to select a majority prediction
#   r: the power term used in the inverse distance weighting formula
#   predFeatures: the features of the prediction set for which we need to provide
#                 labels (must be NULL if cv==TRUE)
#   cv: TRUE if this function is being used for cross-validation (FALSE by default)
#
# For cross-validation:
#   Returns: a scalar in [0,1] representing the accuracy of the algorithm
# Otherwise:
#   Returns a list containing:
#     predLabels: a vector of predicted labels correlating with predFeatures
#     prob: a vector of probabilities corresponding with predLabels
#
# Uses R packages:
#   assertthat (for assertions on parameters)
#
kNN_eval <- function(dist, features, labels, uniqueLabels, k, r, predFeatures=NULL, cv=FALSE)
{
  if (cv)
  {
    assert_that(is.null(predFeatures))
    predFeatures = features
  }
  
  # Reference the required libraries.
  library(assertthat, quietly=TRUE)

  # Assert that the training features, training labels, and prediction features
  # are consistent with each other and have the necessary dimensionality.
  not_empty(features)
  not_empty(labels)
  not_empty(predFeatures)
  assert_that(ncol(features) == ncol(predFeatures))
  assert_that(nrow(features) == length(labels))

  # Sort the distances in the distance matrix by set of test features (columns), extract
  # the indices of those distances, and then throw away all but the first k of each.
  # When doing cross-validation, skip the absolute nearest neighbor.
  kNeighbors <- apply(dist, 1, order)[(1+cv):(k+cv),]

  # Create a matrix of labels with appropriate dimension.
  kLabels <- matrix(NA, k, nrow(predFeatures))
  kDistances <- matrix(NA, k, nrow(predFeatures))
  
  # Extract labels and distances based on the neighbors indexes.  Note that k==1
  # is handled slightly different for syntactical reasons only.
  if (k == 1)
  {
    kLabels[1,] <- labels[kNeighbors]
    kDistances[1,] <- dist[1,kNeighbors]
  }
  else
  {
    kLabels[,] <- labels[kNeighbors]

    for (j in 1:k)
    {
       kDistances[j,] <- dist[,kNeighbors[j]]
    }
  }

  # Create slots for each new prediction.
  predLabels <- rep(0,nrow(predFeatures))
  
  # Loop through all needed predictions, creating buckets for each unique label.
  # Increment the score (the inverse distance weighted value) in each bucket for each
  # of the k nearest neighbors.  Then choose the bucket with the highest score.
  for (m in 1:nrow(predFeatures))
  {
    buckets <- rep(0, length(uniqueLabels))
    for (j in 1:k)
    {
      position <- which(uniqueLabels == kLabels[j,m])
      buckets[position] <- buckets[position] + kDistances[j,m]^(1/r)
    }
    predLabels[m] <- uniqueLabels[which.max(buckets)]
  }

  # If we are doing cross-validation, generate our accuracy and return it.
  if (cv)
  {
    return(mean(predLabels==labels))
  }
  
  # Determine which of the k neighbors match the chosen label...
  matches <- apply(kLabels, 1, `==`, as.matrix(predLabels))
  # ...and then count them.
  counts <- apply(matches, 1, sum)
  # Use the counts to determine our probability scores.
  prob <- counts/k

  # Return predicted labels and corresponding probabilities in a list.
  return(list(predLabels=predLabels, prob=prob))
}

# Function kNN_cv()
#
# Parameters: none
#
# Performs cross-validation for tuning hyperparameters to the kNN_dist and kNN_eval functions.
# Writes results to a CSV.  Prints results as it progresses through the cases.
#
# Returns: nothing
#
# An example of using this function:
# kNN_cv(train, cl, c(4,4.1,4.2,4.3,4.4), c(1,3), c(-0.1,-0.25,-0.5), "MNIST_cv.csv")
#
kNN_cv <- function(trainFeatures, trainLabels, uniqueLabels, pChoices, kChoices, rChoices, outputCsv)
{
  df <- data.frame(k=0, p=0, r=0, result=NA)
  
  for (p in pChoices)
  {
    # The distance matrix only varies based on values of p.  It can be reused among
    # all validation on k and r.
    dist <- kNN_dist(trainFeatures, p, NULL, cv=TRUE)
    for (k in kChoices)
    {
      for (r in rChoices)
      {
        # Reuse the distance matrix and finish the validation with a call to kNN_eval.
        result <- kNN_eval(dist, trainFeatures, trainLabels, uniqueLabels, k, r, NULL, cv=TRUE)
        df <- rbind(df, c(k, p, r, result))
        print(df)
        write.csv(df, outputCsv)
      }
    }
  }
}

# Function kNN_predict() performs kNN predictions based on a training set.
# From the k nearest neighbors, it applies an inverse distance weighting
# formula.  It simply combines calls to kNN_dist and kNN_eval to do its work.
#
# Parameters:
#   trainFeatures: the two-dimensional features from the training set
#   trainLabels: the vector of the labels from the training set
#   p: the 'power' used to define the distance method
#       1=Manhattan distance
#       2=Euclidian distance
#       Inf=Chebyshev distance
#       any other positive number=similar to Euclidian but using power other than 2.
#   k: the number of nearest neighbors from which to select a majority prediction
#   r: the power term used in the inverse distance weighting formula
#   predFeatures: the features of the prediction set for which we need to predict labels
#
# Returns a list containing:
#   predLabels: a vector of predicted labels correlating with predFeatures
#   prob: a vector of probabilities corresponding with predLabels
#
kNN_predict <- function(trainFeatures, trainLabels, uniqueLabels, p, k, r, predFeatures)
{
  dist <- kNN_dist(trainFeatures, p, predFeatures)
  kNN_eval(dist, trainFeatures, trainLabels, uniqueLabels, k, r, predFeatures)
}

# Load the training and test data.
trainDigits <- read.csv("MNIST_training.csv")
trainFeatures <- trainDigits[,2:257]
trainLabels <- trainDigits[,1]
predFeatures <- read.csv("MNIST_test.csv")

# Run the kNN prediction using my tuned parameters for p, k, and r.
results <- kNN_predict(trainFeatures, trainLabels, 0:9, 4.2, 1, -0.25, predFeatures)

# Save the predictions to file.
df <- data.frame(predLabels=results$predLabels)
write.csv(df, "MNIST_predictions.csv", row.names=FALSE)
