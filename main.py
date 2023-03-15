import np as np
import openpyxl
import tkinter.filedialog
import statistics
import scipy

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






#import mpl_interactions.ipyplot as iplt
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






"""

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
"""


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


# Collect Sample Data
proteinName_mapped_to_p_value = {}
proteinName_mapped_to_mean = {}

for row in dataFrameReader.iter_rows(2, dataFrameReader.max_row):
    proteinName = row[2].value
    sampleData[proteinName] = []

    numSamples_coV = 0
    numSamples_noCoV = 0

    coV_mean = 0
    noCoV_mean = 0

    for col in range(5, dataFrameReader.max_column): # 6 = start of non-control groups
        value = row[col].value
        if not isinstance(value, str):
            sampleData[proteinName].append(value)

            if col >= 13: # fixed value for now
                numSamples_coV += 1
            else:
                numSamples_noCoV += 1
        else:
            sampleData[proteinName].append(0)


    if numSamples_coV != 0 and numSamples_noCoV != 0:
        coV_mean = sum(sampleData[proteinName][5:13]) / numSamples_coV
        noCoV_mean = sum(sampleData[proteinName][13:]) / numSamples_noCoV

        res = scipy.stats.ttest_ind(a=sampleData[proteinName][5:13], b=sampleData[proteinName][13:], equal_var=True)
        proteinName_mapped_to_p_value[proteinName] = res.pvalue
        proteinName_mapped_to_mean[
            proteinName] = "Non-Covid mean: " + str(noCoV_mean) + " | Covid mean: " + str(coV_mean) + " || p-value: " + str(res.pvalue)
    else:
        proteinName_mapped_to_p_value[proteinName] = 0.99
        proteinName_mapped_to_mean[proteinName] = "Insufficient Data for statistical significance"

    rowCounter += 1



# Get Sample Names
sampleDataListSampleNames = []
colorMap = {}
for row in dataFrameReader.iter_rows(0, 1):
    for col in range(5, dataFrameReader.max_column):
        s = str(row[col].value)
        #s = s.replace(".PG.Quantity", "")
        s = s.split('_')[0]
        sampleDataListSampleNames.append(s)

        if "noCov" in s:
            colorMap[s] = "blue"
        else:
            colorMap[s] = "red"

sampleDataListTestNumbers = [i for i in range(1,16)]

#print(sampleData)


proteinSampleDataMappedToMean = {}
#desired_p_value = (float)(input("Enter the desired p value: "))
desired_p_value = 0.05
#graphs = input("Would you like to see a graphical display of the statistics for statistically significant proteins? (y/n)")
graphs = "y"

temp_counter = 0
for key,val in sampleData.items():
    # If entire data set is invalid (i.e. "Filtered) then ignore this sample
    #if temp_counter >= 1:
       # break
    #temp_counter += 1

    if len(val) <= 0:
        continue

    proteinSampleDataMappedToMean[key] = []
    proteinSampleDataMappedToMean[key].append(statistics.mean(val))


    #proteinSampleDataMappedToMean[key].append(pvalue_101(graphs, key, desired_p_value, controlDataMappedToMean[key], (int)(proteinSampleDataMappedToMean[key][0]), 100))


    # proteinSampleDataMappedToMean[key].append(proteinMeanGraph(key, controlDataMappedToMean[key])) # Note: this allows GRAPHING


























print("-----------------------------------------------------------")



proteinNamesList = list(sampleData.keys())

from dash import Dash, dcc, html, Input, Output
import plotly.express as px

app = Dash(__name__)


app.layout = html.Div([
    html.H2('Covid vs. Non-Covid Protein Expression Application || Property of the Gong Laboratory.'),
    dcc.Dropdown(
        id="dropdown",
        options=[x for x in sampleData.keys()],
        value=proteinNamesList[0],
        clearable=False,
    ),
    dcc.Graph(id="graph"),
    html.Label("Include results only from proteins with a p-value of less than or equal to: "),
    dcc.Slider(0, 1, marks=None, value=0.05, id='slider', tooltip={"placement": "bottom", "always_visible": True}),
    html.Label("Manually enter a p-value: "),
    dcc.Input(type='number', value=0.05, id='manualinput')
])


@app.callback(
    Output("graph", "figure"),
    Input("dropdown", "value"))
def update_bar_chart(proteinName):
    print(proteinName + ": " + proteinName_mapped_to_mean[proteinName])
    df = {'Samples': sampleDataListSampleNames, 'Detection Levels': sampleData[proteinName]}
    fig = px.bar(df, x = 'Samples', y='Detection Levels',
                 color="Samples",
                 color_discrete_map=colorMap)
    return fig

@app.callback(
    Output("dropdown", "options"),
    Input("slider", "value"))
def update_dropdown(pVal):
    return [key for key,val in proteinName_mapped_to_p_value.items() if val <= pVal]


@app.callback(
    Output("slider", "value"),
    Input("manualinput", "value"))
def update_proteins(pVal):
    if pVal is None:
        return 0
    return pVal

app.run_server(debug=True)
























"""
app = dash.Dash(__name__)

# ---------------------------------------------------------------

df = pd.read_csv(
    "C:\\Users\\antho\\Downloads\\DOHMH_New_York_City_Restaurant_Inspection_Results.csv")  # https://drive.google.com/file/d/1jyvSiRjaNIeOCP59dUFQuZ0_N_StiQOr/view?usp=sharing
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
                              proteinSampleDataMappedToMean.keys()],
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

"""










"""
app = dash.Dash(__name__)

# ---------------------------------------------------------------

df = pd.read_csv(
    "C:\\Users\\antho\\Downloads\\DOHMH_New_York_City_Restaurant_Inspection_Results.csv")  # https://drive.google.com/file/d/1jyvSiRjaNIeOCP59dUFQuZ0_N_StiQOr/view?usp=sharing
df['INSPECTION DATE'] = pd.to_datetime(df['INSPECTION DATE'])
df = df.groupby(['INSPECTION DATE', 'CUISINE DESCRIPTION', 'CAMIS'], as_index=False)['SCORE'].mean()
df = df.set_index('INSPECTION DATE')
df = df.loc['2016-01-01':'2019-12-31']
df = df.groupby([pd.Grouper(freq="M"), 'CUISINE DESCRIPTION'])['SCORE'].mean().reset_index()
# print (df[:5])

df = px.data.iris()





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
                              proteinSampleDataMappedToMean.keys()],
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


   # dff = df[(df['CUISINE DESCRIPTION'] == first_cuisine) |
             #(df['CUISINE DESCRIPTION'] == second_cuisine) |
             #(df['CUISINE DESCRIPTION'] == third_cuisine)]

    fig = px.scatter(df, x='sepal_length', y='sepal_width', color='species', size='petal_length')
    return fig


# ---------------------------------------------------------------

if __name__ == '__main__':
    app.run_server(debug=False)
"""




"""
from dash import Dash, dcc, html
from base64 import b64encode
import io

app = Dash(__name__)

buffer = io.StringIO()

df = px.data.iris() # replace with your own data source
fig = px.scatter(
    df, x="sepal_width", y="sepal_length",
    color="species")
fig.write_html(buffer)

html_bytes = buffer.getvalue().encode()
encoded = b64encode(html_bytes).decode()

app.layout = html.Div([
    html.H4('Simple plot export options'),
    html.P("↓↓↓ try downloading the plot as PNG ↓↓↓", style={"text-align": "right", "font-weight": "bold"}),
    dcc.Graph(id="graph", figure=fig),
    html.A(
        html.Button("Download as HTML"),
        id="download",
        href="data:text/html;base64," + encoded,
        download="plotly_graph.html"
    )
])


app.run_server(debug=True)
"""




