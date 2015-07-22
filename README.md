# sgd
Some implementations / applications of Stochastic Gradient Descent.

## k-means
I've implemented classic k-means and an online / SGD version that makes a single pass through the data.

## Perceptron
I've implemented a perceptron batched model builder. By setting the batch size to 1 you get an SGD perceptron builder; by setting it to the size of the dataset you get the classical gradient descent variation.

# To Run
`sbt run`
Then pick which of the three apps you want to run. The perceptron one currently uses a stochastic perceptron builder.
