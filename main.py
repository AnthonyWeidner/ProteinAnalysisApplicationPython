import np as np
import openpyxl
import tkinter.filedialog
import statistics

#from matplotlib import pyplot as plt

#from matplotlib.widgets import Button
#import mpl_interactions.ipyplot as iplt





import pandas as pd  # (version 1.0.0)
import plotly  # (version 4.5.0)
import plotly.express as px

import dash  # (version 1.8.0)
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State

app = dash.Dash(__name__)

# ---------------------------------------------------------------

df = pd.read_csv(
    "/Users/maggiechen/Downloads/DOHMH_New_York_City_Restaurant_Inspection_Results.csv")  # https://drive.google.com/file/d/1jyvSiRjaNIeOCP59dUFQuZ0_N_StiQOr/view?usp=sharing
df['INSPECTION DATE'] = pd.to_datetime(df['INSPECTION DATE'])
df = df.groupby(['INSPECTION DATE', 'CUISINE DESCRIPTION', 'CAMIS'], as_index=False)['SCORE'].mean()
df = df.set_index('INSPECTION DATE')
df = df.loc['2016-01-01':'2019-12-31']
df = df.groupby([pd.Grouper(freq="M"), 'CUISINE DESCRIPTION'])['SCORE'].mean().reset_index()
# print (df[:5])

# ---------------------------------------------------------------
app.layout = html.Div([

    html.Div([
        dcc.Graph(id='our_graph')
    ], className='nine columns'),

    html.Div([

        html.Br(),
        html.Label(['Choose 3 Cuisines to Compare:'], style={'font-weight': 'bold', "text-align": "center"}),
        dcc.Dropdown(id='cuisine_one',
                     options=[{'label': x, 'value': x} for x in
                              df.sort_values('CUISINE DESCRIPTION')['CUISINE DESCRIPTION'].unique()],
                     value='African',
                     multi=False,
                     disabled=False,
                     clearable=True,
                     searchable=True,
                     placeholder='Choose Cuisine...',
                     className='form-dropdown',
                     style={'width': "90%"},
                     persistence='string',
                     persistence_type='memory'),

        dcc.Dropdown(id='cuisine_two',
                     options=[{'label': x, 'value': x} for x in
                              df.sort_values('CUISINE DESCRIPTION')['CUISINE DESCRIPTION'].unique()],
                     value='Asian',
                     multi=False,
                     clearable=False,
                     persistence='string',
                     persistence_type='session'),

        dcc.Dropdown(id='cuisine_three',
                     options=[{'label': x, 'value': x} for x in
                              df.sort_values('CUISINE DESCRIPTION')['CUISINE DESCRIPTION'].unique()],
                     value='Donuts',
                     multi=False,
                     clearable=False,
                     persistence='string',
                     persistence_type='local'),

    ], className='three columns'),

])


# ---------------------------------------------------------------

@app.callback(
    Output('our_graph', 'figure'),
    [Input('cuisine_one', 'value'),
     Input('cuisine_two', 'value'),
     Input('cuisine_three', 'value')]
)
def build_graph(first_cuisine, second_cuisine, third_cuisine):
    dff = df[(df['CUISINE DESCRIPTION'] == first_cuisine) |
             (df['CUISINE DESCRIPTION'] == second_cuisine) |
             (df['CUISINE DESCRIPTION'] == third_cuisine)]
    # print(dff[:5])

    fig = px.line(dff, x="INSPECTION DATE", y="SCORE", color='CUISINE DESCRIPTION', height=600)
    fig.update_layout(yaxis={'title': 'NEGATIVE POINT'},
                      title={'text': 'Restaurant Inspections in NYC',
                             'font': {'size': 28}, 'x': 0.5, 'xanchor': 'center'})
    return fig


# ---------------------------------------------------------------

if __name__ == '__main__':
    app.run_server(debug=False)




















import mpl_interactions.ipyplot as iplt
import matplotlib.pyplot as plt
import numpy as np

#x = np.linspace(0, np.pi, 100)
#tau = np.linspace(0.5, 10, 100)

#def f1(x, tau, beta):
#    return np.sin(x * tau) * x * beta
#def f2(x, tau, beta):
#    return np.sin(x * beta) * x * tau


#fig, ax = plt.subplots()
#controls = iplt.plot(x, f1, tau=tau, beta=(1, 10, 100), label="f1")
#iplt.plot(x, f2, controls=controls, label="f2")
#_ = plt.legend()
#plt.show()



x = np.linspace(0, np.pi, 100)
p_val = np.linspace(0,10,100)

def graph1(x, p_val, mu):
    return x * p_val * mu


# ACCEPTANCE CRITERA
# AC01: if p value, have a list of proteins fitting the p value
# AC02: bar graph of each individual sample to see variation
# AC03: add typable p value


# NEW FUNCTIONALITY
def proteinMeanGraph(protein_name, mu):
    fig, ax = plt.subplots()
    controls = iplt.plot(x, graph1, p_val=p_val, mu = mu, label=protein_name)
    #iplt.plot(x, graph1, controls=controls)

    _ = plt.legend()

    plt.show()


####################







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

    #proteinSampleDataMappedToMean[key].append(pvalue_101(graphs, key, desired_p_value, controlDataMappedToMean[key], (int)(proteinSampleDataMappedToMean[key][0]), 100))
    proteinSampleDataMappedToMean[key].append(proteinMeanGraph(key, controlDataMappedToMean[key]))

#print(pvalue_101(controlDataMap["Immunoglobulin lambda variable 3-9"], 20.0, (int)(proteinSampleDataMappedToMean["Immunoglobulin lambda variable 3-9"][0]), 33.0))