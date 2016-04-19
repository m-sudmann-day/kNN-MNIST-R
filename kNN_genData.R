#
# kNN
#
# Matthew Sudmann-Day
#
# Applies my k-Nearest Neighbor function kNN() to Nicholas' waveform training set.
#
# Uses R packages:
#   ggplot2

# Nicholas' function for generating waveform data.  
get_data <- function(slice, seed = 12345){
  set.seed(seed)
  x <- seq(0,2*pi,slice)
  y <- runif(length(x)) + sin(x)
  z <- runif(length(x)) + cos(x)
  data <- data.frame(x1 = rep(x,2), x2 = c(y,z), 
                     y = c(rep(0,length(y)), rep(1,length(z)) ))
  return(data)
}

# Create the training data.
trainData <- get_data(0.01, seed=11111)

# Create the prediction data with a different seed.  Note the actual labels go unused.
predictData <- get_data(0.01, seed=22222)

# Run my kNN function to generate the predicted labels using k=3 and p=2.
kNNResult <- kNN(trainData[,1:2], trainData$y, 3, 2, predictData[,1:2])

# Copy results from the kNN function into the prediction data set.
predictData$predLabels <- kNNResult$predLabels
predictData$prob <- kNNResult$prob

# Compare actual and predicted labels.
correctness <- round(mean(predictData$predLabels == predictData$y) * 100)

# Write out the predictions to a CSV.
write.csv(predictData, "predictions.csv", row.names=FALSE)

x1min <- min(predictData$x1)
x1max <- max(predictData$x1)
x2min <- min(predictData$x2)
x2max <- max(predictData$x2)
slice <- 0.05
x1range <- seq(from = x1min - slice, to = x1max + slice, by = slice)
x2range <- seq(from = x2min - slice, to = x2max + slice, by = slice)
x1length <- length(x1range)
x2length <- length(x2range)
backgroundPoints <- data.frame(x1=rep(x1range, x2length), x2=rep(x2range, x1length))
backgroundPoints$y <- as.character(kNN(trainData[,1:2], trainData$y, 3, 2, backgroundPoints)$predLabels)
head(predictData)

# Nicholas' function for creating a plot and writing it to a PDF.
# I modified this function to add contour lines and a title.
save_pdf <- function(data, backgroundPoints, title)
{
  library(ggplot2)
  
  plot <- ggplot(data = data) +
    ggtitle(title) +
    geom_point(aes(x=x1, y=x2, colour=factor(predLabels), shape=factor(y))) +
    #geom_contour(data=backgroundPoints, aes(x=x1, y=x2, z=factor(y)), bins=2) +
    scale_shape_discrete(name="Actual (Shape)") +
    scale_colour_discrete(name="Prediction (Color)") +
    theme_bw()
  ggsave("plot.pdf")
  plot
}

# Save a PDF with a plot containing prediction data and contours of predictions.
save_pdf(predictData,
         backgroundPoints,
         paste("Predicted Labels with Contours (correctness = ", correctness, "%)", sep=""))

