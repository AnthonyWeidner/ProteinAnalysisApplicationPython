import np as np
import openpyxl
import tkinter.filedialog
import statistics

from matplotlib import pyplot as plt


# @params:
# choice: whether user wants to display graphical representation of results or not
# protein_name: name of protein
# desired_p_val: the value that the user entered for the desired p-value (i.e. 0.05, 0.01)
# mu: population mean


# TODO: improve p-value function
def pvalue_101(choice, protein_name, desired_p_val, mu, sigma, samp_size, samp_mean=0, deltam=0):
    np.random.seed(1234)
    s1 = np.random.normal(mu, sigma, samp_size)  # simulate random normal distribution
    if samp_mean > 0:
        print(len(s1[s1>samp_mean]))
        outliers = float(len(s1[s1>samp_mean])*100)/float(len(s1))
        print('Percentage of numbers larger than {} is {}%'.format(samp_mean, outliers))
    if deltam == 0:
        deltam = abs(mu-samp_mean)
    if deltam > 0 :
        outliers = (float(len(s1[s1>(mu+deltam)]))
                    +float(len(s1[s1<(mu-deltam)])))*100.0/float(len(s1))
        #print('Percentage of numbers further than the population mean of {} by +/-{} is {}% for {}'.format(mu, deltam, outliers, protein_name))
        print('Percentage of numbers significantly larger than the control mean is {}% for {}'.format(outliers, protein_name))

    percentage_needed = 1 - desired_p_val
    if outliers < percentage_needed and choice == "y" and samp_size > 5:
        fig, ax = plt.subplots(figsize=(8,8))
        #fig.suptitle('Normal Distribution: population_mean={}'.format(mu) )
        fig.suptitle('Normal Distribution: population_mean for {}'.format(protein_name))
        plt.hist(s1)
        plt.axvline(x=sigma, color='red')
        #plt.axvline(x=mu-deltam, color='green')
        plt.show()

    return outliers


# Set-up program
# File selection
root = tkinter.Tk()
root.withdraw
fileName1 = tkinter.filedialog.askopenfilename()

dataFrame = openpyxl.load_workbook(fileName1)

dataFrameReader = dataFrame.active

# Creating the data structures
arr = [] # goes down each column
sampleData = {} # goes across each row


sampleName = ""
flag = True
counter = 0

for col in range(5, dataFrameReader.max_column):
    arr.append([])
    for row in dataFrameReader.iter_rows(0, dataFrameReader.max_row):
        if flag:
            sampleName = row[col].value
            flag = False

        value = row[col].value
        if not isinstance(value, str):
            arr[counter].append(value)

    arr[counter].sort()
    print("Number of proteins with valid data in", sampleName, "=", len(arr[counter]))
    counter += 1
    flag = True

rowCounter = 0


# Set-up control values
# Goal: parse through just the control column and store data in dictionary
controlDataMap = {}
for row in dataFrameReader.iter_rows(2, dataFrameReader.max_row):
    proteinName = row[2].value
    controlDataMap[proteinName] = []
    for col in range(5, 13):
        value = row[col].value
        if not isinstance(value, str):  # Filtered as value
            controlDataMap[proteinName].append(value)
        else:
            controlDataMap[proteinName].append(50)


controlDataMappedToMean = {}
for key,val in controlDataMap.items():
    if len(val) <= 0:
        continue
    mean = statistics.mean(val)
    controlDataMappedToMean[key] = mean

# get data from left --> right
# key = proteinName
# value = list of data points
for row in dataFrameReader.iter_rows(2, dataFrameReader.max_row):
    proteinName = row[2].value
    sampleData[proteinName] = []
    for col in range(13, dataFrameReader.max_column): # 6 = start of non-control groups
        value = row[col].value
        if not isinstance(value, str):
            sampleData[proteinName].append(value)

    rowCounter += 1


proteinSampleDataMappedToMean = {}
desired_p_value = (float)(input("Enter the desired p value: "))
graphs = input("Would you like to see a graphical display of the statistics for statistically significant proteins? (y/n)")

for key,val in sampleData.items():
    # If entire data set is invalid (i.e. "Filtered) then ignore this sample
    if len(val) <= 0:
        continue

    proteinSampleDataMappedToMean[key] = []
    proteinSampleDataMappedToMean[key].append(statistics.mean(val))

    proteinSampleDataMappedToMean[key].append(pvalue_101(graphs, key, desired_p_value, controlDataMappedToMean[key], (int)(proteinSampleDataMappedToMean[key][0]), 100))

#print(pvalue_101(controlDataMap["Immunoglobulin lambda variable 3-9"], 20.0, (int)(proteinSampleDataMappedToMean["Immunoglobulin lambda variable 3-9"][0]), 33.0))