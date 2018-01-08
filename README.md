# Python-Stock-Market-Prediction-Neural-Network
Python implementation of a multi-layer neural network, working on thousands of training examples. (For Warsaw University of Technology). Prediction is calculated using keras, numpy, scikit, as well as other Python libs.

# How to run
1. Download CSV file from AlphaVantage (https://www.alphavantage.co/)
2. Then run src/main.py and pass it the file, with desired command line arguments
3. See the beautiful results

# Command line arguments for main.py

```-f OR --file```
Path to alpha vantage CSV file to be parsed (Default ../data/daily_MSFT.csv)

```-hc OR --hidden-count```
Number of hidden layer neurons (Default 256)

```-pd OR --previous-days```
Number of previous days used in order to predict the future(Default 14)

```-fd OR --future-days```
For a given amount of previous days predict a sequence of this many future days(Default 3)

```-plot OR --plotted-feature```
The stock feature to be plotted. Currently supported: Open, High, Low, or Close (Default Open)

```-trp OR --training-percentage```
Percentage of data from CSV that will be used as training data (Default 80)

```-b OR --bias```
Bias term to be prepended to concatenated to input matrix (Default 1)


```-k OR --use-keras```
Value can be True or False, defines whether or not we wish to use Keras model. If false we use ELM only, if true we use both ELM and Keras model. (Default False)
