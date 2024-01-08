########################################################### LIBRARIES ###################################################################

# Import all necessary libraries for full code functionality
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
from scipy.signal import medfilt, savgol_filter
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import scipy.signal

########################################################### FUNCTIONS ###################################################################

def filterSignal(signal):
    """ This function is a simple median filter used to partially reduce noise """
    filteredSignal = medfilt(signal, kernel_size=3)
    return filteredSignal

def butterworthFilter(signal, cutoffFrequency, filterType='low', fs=25000, order=2):
    """ This function is a Butterworth low and high pass filter, where the high pass filter is used to normalise the test data before peak 
    detection. The low pass filter used to reduce noise was ultimately replaced by the Savitzky Golay filter. """

    # Make sure a valid filter is selected
    if filterType not in ['low', 'high']:
        raise ValueError("Invalid filter_type. Supported types are 'low' and 'high'.")

    # Use scipy library to design the Butterworth filter
    b, a = scipy.signal.butter(N=order, Wn=cutoffFrequency / (0.5 * fs), btype=filterType, analog=False)

    # Apply the filter to the inputted signal
    filteredSignal = scipy.signal.filtfilt(b, a, signal)

    return filteredSignal

def savGolFilter(signal, windowSize=20, polyOrder=3):
    """ This function is a typical Savitzky Golay digital filter using the scipy signal library. The polynomial order stayed at 3, 
    however, the window size increased for each dataset as follows: D2 and D3: 20, D4: 30, D5 and D6: 50. This increase in window size 
    resulted in a greater degree of smoothing to counteract the increase in noise. """

    # Apply the Savitzky-Golay filter
    smoothedSignal = savgol_filter(signal, windowSize, polyOrder)
    return smoothedSignal

def detectSpikes(filteredSignal, rawSignal):
    """ This function is responsible for detecting spikes in the data, as well as setting the start of these spikes as the Spike Index.
    The function uses a dynamic threshold that can be calibrated for each dataset as detailed within the code. The peaks are then post 
    processed and the start index is found by using an adaptive window that is proportional to the amplitude of the spike, within this 
    window the gradient of the signal is calculated and when there is a sudden change in gradient the function knows the start of the 
    spike has been found. This value is outputted to the start indices vector, along with the threshold value being outputted for 
    debugging purposes. """

    # Set threshold fraction and segment size parameters for threshold calculation
    """ The values below are calibrated to the D2 dataset. For the alternative datasets, the following values should be used:
    D3: Threshold Fraction = 1.3
        Segment Size = 5,000
    D4: Threshold Fraction = 1.38
        Segment Size = 7,000
    D5: Threshold Fraction = 1.66
        Segment Size = 2,000
    D6: Threshold Fraction = 1.96
        Segment Size = 6,000    
    """
    # D2 calibrated paramters
    thresholdFraction = 1.1 
    segmentSize = 6000  

    # Initialise thresholds and calculate the dynamic threshold for spike detection using mean of each segment, added to the product 
    # of the threshold fraction and the standard deviation
    thresholds = []
    for i in range(0, len(rawSignal), segmentSize):
        segment = filteredSignal[i:i + segmentSize]

        threshold = np.mean(segment) + thresholdFraction * np.std(segment)
        
        # Create a lower threshold limit, in case the average mean is zero, accidentally lowering the threshold to zero and
        # making the function ultra-sensitive to spikes. 
        """ The value below is calibrated for the D2 dataset. For the alternative datasets, the following values should be used:
        D3: threshold < 0.44
        D4: threshold < 0.7
        D5: threshold < 1.73
        D6: threshold < 1.4
        """
        if threshold < 0.35:
            threshold = 0.35
        
        thresholds.extend([threshold] * len(segment))

    # Postprocess to keep only the start of each spike
    spikeLabels = (filteredSignal > np.array(thresholds)).astype(int)
    detectedSpikeIndices = np.where(np.diff(spikeLabels) > 0)[0]

    # Set an adaptive window size for gradient change detection, these parameters stayed constant for all datasets.
    amplitudeFraction = 0.02 
    gradientThreshold = 0.2 

    # Calculate the gradient of the signal
    gradient = np.gradient(rawSignal)

    # Initialise start indices and find the points where the gradient changes significantly within the detected spike window
    startIndices = []
    for i in detectedSpikeIndices:
        amplitudeBasedWindow = int(amplitudeFraction * np.abs(filteredSignal[i]))
        windowSize = max(5, amplitudeBasedWindow)  # Ensure a minimum window size
        windowStart = max(0, i - windowSize)
        windowEnd = min(len(filteredSignal), i + windowSize)
        
        # Find the index of the minimum value within the window and set to start indices array
        minIndex = windowStart + np.argmin(filteredSignal[windowStart:windowEnd])
        if np.abs(np.diff(gradient[windowStart:windowEnd])).max() > gradientThreshold:
            startIndices.append(minIndex)
    
    # Debugging print statement to see number of spikes detected
    print("Number of detected spikes:", np.array(startIndices).shape)

    return startIndices, thresholds

def extractWholeSpikeTrain(signal, index, classes, window_size=100):
    """ This function is responsible for extract the whole spikes for the training of KNN model, it takes the index and classes from
    D1 and extracts all the data points within a window of 100 from each index. These whole spikes and corresponding neuron class
    are then outputted. """
    # Initialise vectors
    spikes, neurons = [], []

    for startIndex in index:
        # Calculate the window end based on the fixed window size and extract values
        window_end = startIndex + window_size
        wholeSpikes = signal[startIndex:window_end]

        # Find the corresponding location in trainIndex and append the vectors
        location = np.where(trainIndex == startIndex)[0][0]
        spikes.append(wholeSpikes)
        neurons.append(classes[location])

    return spikes, neurons

def extractWholeSpikeTest(signal, startIndices, windowSize=100):
    """ This function extracts the whole spikes for the testing of the KNN model, it takes the start indices that have been detected
    in the dataset and finds the data points in a window of 100 after each one. Before appending the spikes to the spikes vector, the 
    spikes are padded / trimmed depending on their size to the specified window size. This ensures no mis-match of vector shapes. """
    # Initialise vectors
    spikes = []

    for startIndex in startIndices:
        # Calculate the window end based on the fixed window size and extract values
        windowEnd = startIndex + windowSize
        wholeSpikes = signal[startIndex:windowEnd]

        # Pad / trim the spike to fit the given window size
        if len(wholeSpikes) < windowSize:
            # If the spike is shorter than the window size, then pad with zeros
            padLength = windowSize - len(wholeSpikes)
            wholeSpikes = np.pad(wholeSpikes, (0, padLength), 'constant')
        elif len(wholeSpikes) > windowSize:
            # If the spike is longer than the window size, trim the spike
            wholeSpikes = wholeSpikes[:windowSize]

        # Append spike vector with the whole spikes
        spikes.append(wholeSpikes)

    return spikes

def applyPca(X, nComponents=10):
    """ This function performs principal component analysis for dimensionality reduction. This function uses the sklearn library. The 
    number of n components stayed constant for all datasets. """
    pca = PCA(n_components=nComponents)
    X_pca = pca.fit_transform(X)
    return X_pca

def performKNNClassification(X_train, y_train, X_test, n_neighbors=5):
    """ This function is responsible for performing the K-Nearest Neighbours Algorithm to form the classifications of the spike labels
    and neurons. The function takes the training data and performs non-parametric supervised learning."""

    # Create a KNN classifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Train the KNN model
    knn.fit(X_train, y_train)

    # Predict the class for each spike in the test data
    predictedClasses = knn.predict(X_test)

    return predictedClasses

########################################################### RUN CODE ###################################################################

# Import all training data, this code changes for each dataset and has been left with no directory to uphold author anonymity
trainData = spio.loadmat('Insert/Directory/Here/D1.mat', squeeze_me=True)
trainD = trainData['d']
trainIndex = trainData['Index']
trainClass = trainData['Class']

# Import the testing data, this code changes for each dataset and has been left with no directory to uphold author anonymity
testData = spio.loadmat('Insert/Directory/Here/D2/D3/D4/D5/D6.mat', squeeze_me=True)
testD = testData['d']

# Initially filter signal for training with median filter
filteredTrainingSignal = filterSignal(trainD)

# Extract the whole spikes from training data
trainSpikes, trainClasses = extractWholeSpikeTrain(filteredTrainingSignal, trainIndex, trainClass, window_size = 100)

# Apply PCA for dimensionality reduction of data
nComponents = 10 
trainSpikesPCA = applyPca(trainSpikes, nComponents)

# Perform KNN classification on the training data
predictedClassesTraining = performKNNClassification(trainSpikesPCA, trainClasses, trainSpikesPCA)

# Normalise testing data with a butteworth high pass filter
testDataNormalised = butterworthFilter(testD, cutoffFrequency=25, filterType='high', fs=25000, order=2)

# Filter test data to remove noise with median filter and then the Savitzky Golay filter
medianFiltered = filterSignal(testDataNormalised)
filteredTestData = savGolFilter(medianFiltered)

# Use peak detection and local gradients to find start indices for the testing spikes
startIndicesTesting, testThresholds = detectSpikes(filteredTestData, testD)

# Debugging print statements checking the shape of the spike arrays
print("Number of detected spikes:", np.array(startIndicesTesting).shape)
print("Total Spikes needed:", np.array(trainIndex).shape)

# Plot the original D1.mat signal, filtered test data, and the detected start indices with actual index values
# This is used for debugging purposes, allowing for manual calibration of threshold parameters. 
plt.figure(figsize=(12, 6))
plt.plot(testD, label='Original D2.mat Signal', alpha=0.7)
plt.plot(filteredTestData, label='Filtered Test Data')
plt.scatter(startIndicesTesting, filteredTestData[startIndicesTesting], color='red', label='Detected Start Indices')
plt.plot(testThresholds, color='green', linestyle='--', label='Dynamic Threshold')
plt.title('Original D2.mat Signal, Filtered Test Data with Detected Start Indices, Dynamic Threshold, and Original Index Values')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()
plt.show()

# Extract whole spikes from testing data
testSpikes = extractWholeSpikeTest(filteredTestData, startIndicesTesting, windowSize=100)

# Apply PCA for dimensionality reduction on testing data
testSpikesPCA = applyPca(testSpikes, nComponents)

# Perform KNN classification on the testing data
predictedClassesTesting = performKNNClassification(trainSpikesPCA, trainClasses, testSpikesPCA)

# Save both start indices and predicted class to a .mat file, the directory has been removed for author anonymity
resultsFile = "Save/Results/D2/D3/D4/D5/D6.mat"
data_to_save = {'Index': startIndicesTesting, 'Class': predictedClassesTesting}
spio.savemat(resultsFile, data_to_save)

# Print message after saving results
print("Results saved successfully.")
