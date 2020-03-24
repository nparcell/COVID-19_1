import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates 
import csv  
import math
import sys 
from scipy.stats import pearsonr 
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn.svm import SVR
from keras.models import Sequential 
from keras.layers import Dense, LSTM, Flatten, Dropout, Activation, Conv1D
from keras import metrics, callbacks 
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from itertools import cycle 
import seaborn as sns 
sns.set() 
from itertools import cycle 
import datetime 

class Modeling():

    """
    - Compile time-averaged data in to one dataset by fetching from server
    - - Labels - formaldehyde, acetaldehyde concentrations
    
    - Split data in to training and testing data
    - - Normalize or re-scale? I think I have to if building LSTM model
    - Build predictive model
    -   - 1) PLS
    -   - 2) LSTM 
    -   -> Probably compare the results of the two later
    - Find R2 values for each type of predictive method
    """

    def __init__(self, trainData, testData, shuffle, channelsToUse, features, environmentalFeatures):


        self.trainData = trainData 
        self.testData = testData 
        
        # Import data using pandas 
        trainData = pd.read_csv(self.trainData) 
        testData = pd.read_csv(self.testData) 
        self.train_df = pd.DataFrame(trainData) 
        trainData = pd.DataFrame(trainData) 
        testData = pd.DataFrame(testData) 

        # Make NaN's 0's 
        trainData.fillna(0, inplace = True) 
        testData.fillna(0, inplace = True) 

        # Set up X's and Y's 
        self.trainX, self.trainY = [], [] 
        self.testX, self.testY = [], []  
        for column in trainData.columns: 
            # if "CH" in column:  
            for channel in channelsToUse: 
                if channel in column:
                    for feature in features: 
                        if feature in column:
                            self.trainX.append(np.array(trainData[column])) 
                            self.testX.append(np.array(testData[column]))  
            if "Air Sentry" in column: 
                print(column)
                self.trainY.append(np.array(trainData[column])) 
                self.testY.append(np.array(testData[column])) 
            for feature in environmentalFeatures:
                if feature in column:
                    self.trainX.append(np.array(trainData[column])) 
                    self.testX.append(np.array(testData[column]))

        self.trainX = np.array(self.trainX).transpose() 
        self.trainY = np.array(self.trainY).transpose() 
        self.testY = np.array(self.testY).transpose() 
        self.testX = np.array(self.testX).transpose() 

        self.trainX, self.testX = self.transformData(self.trainX), self.transformData(self.testX) 

        # Create PLS data, and make NaN's 0  
        self.XtrainPLS = np.array(self.trainX) 
        self.XtestPLS = np.array(self.testX) 
        self.YtrainPLS = np.array(self.trainY) 

        # Make NaN's 0 
        # np.where(np.isnan(self.XtrainPLS), self.XtrainPLS, 0) 
        # np.where(np.isnan(self.XtestPLS), self.XtestPLS, 0) 
        # np.where(np.isnan(self.YtrainPLS), self.YtrainPLS, 0) 
        # np.nan_to_num(self.XtrainPLS) 
        # np.nan_to_num(self.XtestPLS) 
        # np.nan_to_num(self.YtrainPLS) 

        # Create LSTM data (has to be reshaped) 
        self.trainX_reshaped = np.reshape(self.trainX, (self.trainX.shape[0], 1, self.trainX.shape[1])) 
        self.testX_reshaped = np.reshape(self.testX, (self.testX.shape[0], 1, self.testX.shape[1])) 
        self.Xtrain_LSTM = self.trainX_reshaped[:] 
        self.Xtest_LSTM = self.testX_reshaped[:] 

    def transformData(self, inputVar):

        sc = MinMaxScaler()
        sc.fit(inputVar)
        return sc.transform(inputVar)

    def LSTM(self):

        callback = callbacks.EarlyStopping(
            monitor = "mean_absolute_error",
            # patience = 50,
            # patience = 20, 
            # patience = 10,
            patience = 15, 
        )

        numNeurons = 128
        numEpochs = 1000
        dropoutRate = 0

        model = Sequential()

        # Add 1D Conv layer
        modelNums = [
                    128,
                    64,
                    32,
                    16,
                    8,
                    4
                    ]
        kernel_size = 1
        activation = "relu"
        for num in modelNums:
            model.add(Conv1D(num, (kernel_size),
                            activation = activation,
                            padding = "same"
            ))

        model.add(LSTM(
            # int(numNeurons/2),  
            numNeurons, 
            activation = "tanh",
            recurrent_activation = "hard_sigmoid",
            dropout = dropoutRate, 
            return_sequences = True, 
        )) 
        model.add(LSTM(
            int(numNeurons/2),  
            # numNeurons/2, 
            activation = "tanh",
            recurrent_activation = "hard_sigmoid",
            dropout = dropoutRate,
        )) 
        # model.add(Flatten())
        model.add(Dense(numNeurons)) 
        model.add(Dense(int(numNeurons/2))) 
        model.add(Dense(int(numNeurons/4))) 
        model.add(Dense(8)) 
        model.add(Dense(4)) 
        model.add(Dense(1))
        model.compile(
            loss = "mean_squared_error",
            optimizer = "rmsprop",
            metrics = [metrics.mae]
        )
        model.fit(self.Xtrain_LSTM, self.trainY,
            epochs = numEpochs,
            verbose = 2,
            callbacks = [callback]
        )

        # model.fit(self.Xtrain_LSTM, self.YtrainPLS)
        model.evaluate(self.Xtrain_LSTM, self.YtrainPLS)
        self.LSTMPred_train = model.predict(self.Xtrain_LSTM) 
        self.LSTMPred_test = model.predict(self.Xtest_LSTM)

    def PLS(self):

        pls = PLSRegression(n_components = 2)
        pls.fit(self.XtrainPLS, self.YtrainPLS)
        self.PLS_Y_train = pls.predict(self.XtrainPLS)
        self.PLS_Y_test = pls.predict(self.XtestPLS) 

    def SVR_model(self): 
        """
        Support Vector Regression 
        """  
        svr = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1, coef0=1) 
        # svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1) 
        # svr = SVR(kernel = "sigmoid", C = 100, gamma = "auto", epsilon = 0.1)  
        # svr = SVR(kernel = "precomputed") 
        svr.fit(self.XtrainPLS, self.YtrainPLS) 
        self.SVR_Y_train = svr.predict(self.XtrainPLS) 
        self.SVR_Y_test = svr.predict(self.XtestPLS) 

    def gradientBoost_model(self): 
        """
        Only use this if there are distinct categories, such as:
        - 5ppb
        - 10ppb
        - 15ppb
        - 20ppb 
        """
        gradientBoost = XGBClassifier() 
        gradientBoost.fit(self.XtrainPLS, self.YtrainPLS) 
        self.xgboost_Y_train = gradientBoost.predict(self.XtrainPLS) 
        self.xgboost_Y_test = gradientBoost.predict(self.XtestPLS) 

    def R2Analysis(self):
        """
        This is for when you actually have test results to compare predictions to 
        """
        showY = np.array(self.Ytest).transpose()
        showPLSPreds = np.array(self.PLS_Y).transpose()
        showLSTMPreds = np.array(self.LSTMPred).transpose()

        corrArray = {
                    "PLS R2" : 0,
                    "LSTM R2" : 0,
                    }

        corrArray["PLS R2"], _ = pearsonr(showPLSPreds[0], showY[0])
        corrArray["LSTM R2"], _ = pearsonr(showLSTMPreds[0], showY[0])

        self.corr = corrArray 

    def RMSE(self, actual, predicted):
        """
        Equation for root mean square error 
        """
        summedVal = 0 
        for k, element in enumerate(predicted): 
            val_pred, val_act = element, actual[k] 
            summedVal += (val_pred - val_act) **2 

        N = len(predicted) 
        RMSE = (summedVal / N) ** (1/2) 
        return RMSE 

    def RMSE_analysis(self): 

        print("RMSE analysis") 
        print("Creating weights from RMSE value")

        # First, compute the RMSE's for each model 
        self.RMSE_PLS = self.RMSE(self.trainY, self.PLS_Y_train) 
        self.RMSE_LSTM = self.RMSE(self.trainY, self.LSTMPred_train) 
        self.RMSE_SVR = self.RMSE(self.trainY, self.SVR_Y_train) 
        self.RMSE_xgboost = self.RMSE(self.trainY, self.xgboost_Y_train) 
        print("RMSE - PLS: %0.3f"%self.RMSE_PLS) 
        print("RMSE - LSTM: %0.3f"%self.RMSE_LSTM)  
        print("RMSE - SVR: %0.3f"%self.RMSE_SVR) 
        print("RMSE -xgboost: %0.3f"%self.RMSE_xgboost)

        # Next, going to create some weights from these 
        self.w_PLS = 1 - (self.RMSE_PLS / (self.RMSE_LSTM + self.RMSE_PLS + self.RMSE_SVR + self.RMSE_xgboost)) 
        self.w_LSTM = 1 - (self.RMSE_LSTM / (self.RMSE_LSTM + self.RMSE_PLS + self.RMSE_SVR + self.RMSE_xgboost)) 
        self.w_SVR = 1 - (self.RMSE_SVR / (self.RMSE_LSTM + self.RMSE_PLS + self.RMSE_SVR + self.RMSE_xgboost)) 
        self.w_xgboost = 1 - (self.RMSE_xgboost/ (self.RMSE_LSTM + self.RMSE_PLS + self.RMSE_SVR + self.RMSE_xgboost))
        print("weight - PLS: %0.2f"%self.w_PLS) 
        print("weight - LSTM: %0.2f"%self.w_LSTM) 
        print("weight - SVR: %0.2f"%self.w_SVR) 
        print("weight - xgboost: %0.2f"%self.w_xgboost) 

    def RMSE_after_testing(self): 

        # Let's look at some test data metrics 
        self.RMSE_PLS_test = self.RMSE(self.testY, self.PLS_Y_test) 
        self.RMSE_LSTM_test = self.RMSE(self.testY, self.LSTMPred_test) 
        self.RMSE_SVR_test = self.RMSE(self.testY, self.SVR_Y_test) 
        self.RMSE_xgboost_test = self.RMSE(self.testY, self.xgboost_Y_test)
        self.RMSE_fusion_test = self.RMSE(self.testY, self.fusedTest) 
        print("RMSE - PLS test predictions: %0.3f"%self.RMSE_PLS_test) 
        print("RMSE - LSTM test predictions: %0.3f"%self.RMSE_LSTM_test) 
        print("RMSE - SVR test predictions: %0.3f"%self.RMSE_SVR_test) 
        print("RMSE - xgboost test predictions: %0.3f"%self.RMSE_xgboost_test) 
        print("RMSE - fused test predictions: %0.3f"%self.RMSE_fusion_test)  
    
    def weightedAverage(self, weights, values): 
        """
        weights = list or array of weights 
        values = list or array of values 
        """
        N = len(weights) 
        w_summed = sum(weights) 

        summedVal = 0 
        for k, value in enumerate(values): 
            summedVal += weights[k] * value 

        summedVal = summedVal / (N * w_summed) 
        return summedVal 

    def easyWeighting(self, weights, values): 
        """
        pretty simple weighted addition concept 
        ---
        value = weight1 * value1 + weight2 * value2. 
        -> probably preferable to use w's not RMSE's for this 
        """
        summedVal = 0 
        for k, weight in enumerate(weights): 
            summedVal += weight * values[k] 
        
        return summedVal 

    def weightedGeometricMean(self, weights, values): 
        """
        Found this equation from https://en.wikipedia.org/wiki/Weighted_geometric_mean
        """ 
        w_summed = sum(weights)  
        
        summedVal = 0
        for k, weight in enumerate(weights): 
            summedVal += weight * np.log(values[k]) 

        summedVal  = summedVal / w_summed
        
        return summedVal 
    
    def geometricMean(self, values): 
        """
        Found this equation from https://en.wikipedia.org/wiki/Geometric_mean
        """
        multipliedVal = 1 
        for element in values: 
            multipliedVal *= element 

        N = len(values) 

        try:
            multipliedVal = (multipliedVal) ** (1 / N) 
        except: 
            print(multipliedVal) 

        return multipliedVal 

    def modelFusion(self, style): 

        print("Model fusion using style: ", style) 

        # Now, going to fuse the two predictions together 
        self.fusedTrain = np.zeros(len(self.trainY))  # pre-allocate  
        weights = [ # self.w_X is 1 - percentage, self.RMSE is the raw RMSE value as a weight  
            self.w_PLS, 
            self.w_LSTM, 
            self.w_SVR, 
            self.w_xgboost,
            # self.RMSE_PLS, 
            # self.RMSE_LSTM, 
            # self.RMSE_SVR,
            # self.RMSE_xgboost,
            ]  

        print("Model fusion predictions on train data") 
        values = np.zeros(len(weights))  
        for k, element in enumerate(self.LSTMPred_train): 
            # values = [self.PLS_Y_train[k], element]  
            values[0] = self.PLS_Y_train[k] 
            values[1] = element 
            values[2] = self.SVR_Y_train[k] 
            values[3] = self.xgboost_Y_train[k]
            if style == "weighted average": 
                self.fusedTrain[k] = self.weightedAverage(weights = weights, values = values)  
            elif style == "easy weighting": 
                self.fusedTrain[k] = self.easyWeighting(weights = weights, values = values) 
            elif style == "weighted geometric mean": 
                self.fusedTrain[k] = self.weightedGeometricMean(weights = weights, values = values) 
            elif style == "geometric mean": 
                self.fusedTrain[k] = self.geometricMean(values = values) 
        RMSE_fused_train = self.RMSE(self.trainY, self.fusedTrain) 
        print(np.where(self.fusedTrain == np.nan))
        print("RMSE - Fused:  %0.2f"%RMSE_fused_train) 

        # Now do this fusion for test predictions 
        print("Model fusion predictions on test data") 
        self.fusedTest = np.zeros(len(self.LSTMPred_test)) 
        for k, element in enumerate(self.LSTMPred_test): 
            # values = [self.PLS_Y_test[k], element]  
            values[0] = self.PLS_Y_test[k] 
            values[1] = element 
            values[2] = self.SVR_Y_test[k] 
            values[3] = self.xgboost_Y_test[k] 
            if style == "weighted average": 
                self.fusedTest[k] = self.weightedAverage(weights = weights, values = values)  
            elif style == "easy weighting":
                self.fusedTest[k] = self.easyWeighting(weights = weights, values = values) 
            elif style == "weighted geometric mean": 
                self.fusedTest[k] = self.weightedGeometricMean(weights = weights, values = values) 
            elif style == "geometric mean": 
                self.fusedTest[k] = self.geometricMean(values = values) 

    def justShowPredictions(self, style): 
        """
        This is for when you only have test predictions to show 
        """ 
        # Create datetime array with start date
        # yearStart, monthStart, dayStart = self.date[6:], self.date[0:2], self.date[3:5] 
        # print("year: " + yearStart + " month: " + monthStart + " day: " + dayStart) 
        # yearStart, monthStart, dayStart = int(yearStart), int(monthStart), int(dayStart) 
        # print(yearStart, monthStart, dayStart)

        class colors: 
            LSTM = "g"
            PLS = "b" 
            ACTUAL = "k" 
            FUSED = "pink" 
            SVR = "purple" 
            xgboost = "orange" 

        # Plot predictions 
        plt.figure(2) 
        self.LSTMPred_test = np.array(self.LSTMPred_test).transpose() 
        self.PLS_Y_test = np.array(self.PLS_Y_test).transpose() 
        self.SVR_Y_test = np.array([self.SVR_Y_test]).transpose() 
        lines = ["-", "--", "-.", ":"] 
        linecycler = cycle(lines) 

        plt.title("Test Data - Ammonia (ppbv) predictions", fontsize = 22) 
        plt.plot(self.LSTMPred_test[0], label = "LSTM", linestyle = "-", c = colors.LSTM) 
        plt.plot(self.PLS_Y_test[0], label = "PLS", linestyle = "-", c = colors.PLS) 
        plt.plot(self.SVR_Y_test, label = "SVR", linestyle = "-", c = colors.SVR) 
        plt.plot(self.xgboost_Y_test, label = "xgboost", linestyle = "-", c = colors.xgboost) 
        plt.plot(self.fusedTest, label = "fused with " + style, linestyle = "-", c = colors.FUSED) 
        plt.plot(self.testY, label = "actual", c = colors.ACTUAL) 

        # Create date range and plot
        # locs, xticks = plt.xticks() 
        # Nticks = len(xticks) - 2
        # algebra = len(self.LSTMPred[0]) / Nticks 
        # endDate = str(monthStart) + "-" + str(dayStart + 1) + "-" + str(yearStart)
        # dateRange = pd.date_range(start = self.date, end = endDate, periods = Nticks, 
        # ) 
        # newDateRange = [str(dateRange[j]) for j in range(len(dateRange))] 
        # newDateRange.insert(0, "")  # insert empty at beginning
        # newDateRange.insert(len(newDateRange),"")  # insert empty at end 
        # plt.yticks(fontsize = 18)
        # plt.xticks(locs, newDateRange, rotation = 15, horizontalalignment = "right", fontsize = 18) 

        # plt.xlabel("Time", fon) 
        # plt.xlabel("Time", fontsize = 20)
        plt.ylabel("Concentration (ppbv)", fontsize = 20)
        plt.legend(fontsize = 20) 
        plt.show(block = False) 

        plt.figure(4) 
        testxlin = np.linspace(-2, max(self.testY)) 
        plt.plot(testxlin, testxlin)  
        plt.scatter(self.testY, self.PLS_Y_test[0], label = "PLS", c = colors.PLS) 
        plt.scatter(self.testY, self.SVR_Y_test, label = "SVR", c = colors.SVR) 
        plt.scatter(self.testY, self.fusedTest, label = "Fusion", c = colors.FUSED) 
        plt.scatter(self.testY, self.xgboost_Y_test, label = "xgboost", c = colors.xgboost) 
        plt.scatter(self.testY, self.LSTMPred_test[0], label = "LSTM", c = colors.LSTM) 
        plt.xlabel("Actual", fontsize = 20) 
        plt.ylabel("Predicted", fontsize = 20) 
        plt.legend(fontsize = 20) 
        plt.title("Test Predictions Regression", fontsize = 22) 
        plt.show(block = False) 

        plt.figure(1) 
        self.LSTMPred_train = np.array(self.LSTMPred_train).transpose() 
        self.PLS_Y_train = np.array(self.PLS_Y_train).transpose() 
        self.SVR_Y_train = np.array([self.SVR_Y_train]).transpose() 

        plt.title("Train Data - Ammonia (ppbv) predictions", fontsize = 22) 
        plt.plot(self.trainY, label = "Actual", linestyle = "-", c = colors.ACTUAL) 
        plt.plot(self.LSTMPred_train[0], label = "LSTM", linestyle = "-", c = colors.LSTM) 
        plt.plot(self.PLS_Y_train[0], label = "PLS", linestyle = "-", c = colors.PLS ) 
        plt.plot(self.SVR_Y_train, label = "SVR", linestyle = "-", c = colors.SVR) 
        plt.plot(self.xgboost_Y_train, label = "xgboost", linestyle = "-", c = colors.xgboost)
        plt.plot(self.fusedTrain, label = "fused with " + style, linestyle = "-", c = colors.FUSED)
        plt.ylabel("Concentration (ppbv)", fontsize = 20) 
        plt.legend(fontsize = 20) 
        plt.show(block = False) 

        plt.figure(3) 
        plt.title("Train Data - Regression", fontsize = 22) 
        xlin = np.linspace(-2, max(self.trainY)) 
        plt.plot(xlin, xlin, linestyle = ":") 
        plt.scatter(self.trainY, self.PLS_Y_train[0], label = "PLS", c = colors.PLS) 
        plt.scatter(self.trainY, self.fusedTrain, label = "fused with " + style, c = colors.FUSED) 
        plt.scatter(self.trainY, self.SVR_Y_train, label = "SVR", c = colors.SVR) 
        plt.scatter(self.trainY, self.xgboost_Y_train, label = "xgboost", c = colors.xgboost) 
        plt.scatter(self.trainY, self.LSTMPred_train[0], label = "LSTM", c = colors.LSTM) 
        plt.xlabel("Actual (ppbv)", fontsize = 20) 
        plt.ylabel("Predicted (ppbv)", fontsize = 20) 
        plt.legend(fontsize = 20) 
        plt.show()  

    def showPredictions(self):

        showY = np.array(self.Ytest).transpose()
        showPLSPreds = np.array(self.PLS_Y).transpose()
        showLSTMPreds = np.array(self.LSTMPred).transpose()

        # fig, ax = plt.subplots(2, 2)
        plt.figure(1)
        plt.plot(showY[0], linestyle = "-", label = "Actual")
        plt.plot(showPLSPreds[0], linestyle = "-.", label = "PLS R2 = "+"%0.2f"%(self.corr["PLS Formaldehyde R2"]**2))
        plt.plot(showLSTMPreds[0], linestyle = ":", label = "LSTM R2 = "+"%0.2f"%(self.corr["LSTM Formaldehyde R2"]**2))
        plt.title(self.whichArea + " Form. Comparison", fontsize = 18)
        plt.ylabel("Concentration (ppbv)", fontsize = 15)
        plt.ylim(0, max(showLSTMPreds[0]) + 2)
        plt.axis("square") 
        plt.legend(fontsize = 15)
        plt.show(block = False)

        plt.figure(2)
        plt.plot(showY[1], linestyle = "-", label = "Actual")
        plt.plot(showPLSPreds[1], linestyle = "-.", label = "PLS R2 = "+"%0.2f"%(self.corr["PLS Acetaldehyde R2"]**2))
        plt.plot(showLSTMPreds[1], linestyle = ":", label = "LSTM R2 = "+"%0.2f"%(self.corr["LSTM Acetaldehyde R2"]**2))
        plt.title(self.whichArea + " Acet. Comparison", fontsize = 18)
        plt.ylabel("Concentration (ppbv)", fontsize = 15)       
        plt.ylim(0, max(showLSTMPreds[1]) + 2)
        plt.axis("square") 
        plt.legend(fontsize = 15)
        plt.show(block = False)

        xlinForm, xlinAcetal = np.linspace(0, max(showY[0])+1.5), np.linspace(0, max(showY[1])+1)

        # fig.suptitle(self.whichArea + " Actual vs. Predicted Concentrations (ppbv)", fontsize = 24)
        plt.figure(3)
        plt.plot(xlinForm, xlinForm, c = "k")
        plt.scatter(showY[0], showPLSPreds[0], c = "blue", label = "PLS Pred. R2 = "+"%0.2f"%(self.corr["PLS Formaldehyde R2"]**2))
        plt.scatter(showY[0], showLSTMPreds[0], c = "red", label = "LSTM Pred. R2 = "+"%0.2f"%(self.corr["LSTM Formaldehyde R2"]**2), marker = "^")
        plt.title("Formaldehyde Regression", fontsize = 18)
        plt.ylabel("Predictions", fontsize = 15)
        plt.xlabel("Actual Formaldehyde (ppbv)", fontsize = 15)
        plt.axis("square")
        plt.legend(fontsize = 15)
        plt.show(block = False)

        plt.figure(4)
        plt.plot(xlinAcetal, xlinAcetal, c = "k") 
        plt.scatter(showY[1], showPLSPreds[1], c = "blue", label = "PLS Pred. R2 = " + "%0.2f"%(self.corr["PLS Acetaldehyde R2"]**2))
        plt.scatter(showY[1], showLSTMPreds[1], c = "red", label = "LSTM Pred. R2 = "+"%0.2f"%(self.corr["LSTM Acetaldehyde R2"]**2), marker = "^")
        plt.title("Acetaldehyde Regression", fontsize = 18)
        plt.ylabel("Predictions", fontsize = 15)
        plt.xlabel("Actual Acetaldehyde(ppbv)", fontsize = 15)
        plt.axis("square")
        plt.legend(fontsize = 15)
        # plt.show(block = False)
        plt.show()

if __name__ == "__main__": 
    
    # Change directory to the folder where data is 
    os.chdir("covid19-global-forecasting-week-1")

    # Predictive Modeling
    a = Modeling(
                trainData = "train.csv", 
                testData = "test.csv",
                shuffle = False, 
    )
    a.PLS()
    a.SVR_model() 
    a.gradientBoost_model() 
    a.LSTM() 

    # Some R2 analysis
    # a.createOverallTest(date = date, area = whichArea) 
    # a.R2Analysis()

    # RMSE analysis 
    a.RMSE_analysis() 

    # Model fusion 
    style = "geometric mean"  # weighted average, easy weighting, weighted geometric mean, geometric mean 
    a.modelFusion(
        style = style,  
    ) 

    # Plot test data performance metrics 
    a.RMSE_after_testing()

    # Plotting
    a.justShowPredictions(style = style) 

    # Datasets to be used 
    datasets = [
        "train.csv", 
        "test.csv", 
    ]

    # Algorithm 