# NeuronSpikeMachineLearning
System to automatically analyse a set of recordings from a human brain, detecting and classifying neuron spikes through machine learning in Python.

Using various Computational Intelligence methods, a spike detection programme was successfully designed to find the start of neuron spikes from a simple bipolar electrode inserted within the cortical region of the brain. The code starts by importing all the D1.mat training data, and the relevant testing data from D2-D6.mat. It then applies a median filter to the training data. A function is then called to extract the whole spikes for the training of KNN model, it takes the index and classes from D1.mat and extracts all the data points within a window of 100 from each index. These whole spikes and corresponding neuron class are then outputted. A Principal Component Analysis (PCA) function is then called for dimensionality reduction of the training data, this uses the sklearn library. Following this, a function is called to perform the K-Nearest Neighbours Algorithm to form the classifications of the spike labels and neurons. This function takes the training data and performs non-parametric supervised learning. 

The test data is first normalised using a Butterworth high-pass filter, and then noise is reduced using a median filter followed by a Savitzky–Golay filter. This filter uses the scipy signal library. A spike detection function is then called, this function is responsible for detecting spikes in the data, as well as setting the start of these spikes as the Spike Index. The function uses a dynamic threshold that was calibrated for each dataset. The peaks are then post processed and the start index is found by using an adaptive window that is proportional to the amplitude of the spike. Within this window, the gradient of the signal is calculated and when there is a sudden change in gradient the function knows the start of the spike has been found. This value is then outputted to the start indices vector. Following this, the whole spikes are extracted from the testing data, this operates in a similar manner to the spike extraction for the training data, however, also includes padding and trimming to ensure consistent window sizes. The PCA function is then called again, along with the KNN classification function. However, this time, the KNN predicts the neuron classes. The start indices and predicted classes are then saved to a D2/D3/D4/D5/D6.mat file, depending on the dataset being tested. This resulted in precision and recall values ranging from highs of 97% in D2 to lows of 75% in D6.
