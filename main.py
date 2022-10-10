import tkinter
import openpyxl
import tkinter.filedialog

# importing the required modules
#from tkinter import *                   # importing all the widgets and modules from tkinter
#from tkinter import messagebox as mb    # importing the messagebox module from tkinter
#from tkinter import filedialog as fd    # importing the filedialog module from tkinter
#import os                               # importing the os module
#import shutil                           # importing the shutil module

root = tkinter.Tk()
root.withdraw
fileName1 = tkinter.filedialog.askopenfilename()

# ----------------- defining functions -----------------
# function to open a file
#def openFile():
   # selecting the file using the askopenfilename() method of filedialog
 #  the_file = fd.askopenfilename(
  #    title = "Select a file of any type",
   #   filetypes = [("All files", "*.*")]
    #  )
   # opening a file using the startfile() method of the os module
   #os.startfile(os.path.abspath(the_file))

#openFile()
dataFrame = openpyxl.load_workbook(fileName1)

dataFrameReader = dataFrame.active
arr = []

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
    #print(arr)
    print("Number of proteins with valid data in", sampleName, "=", len(arr[counter]))
    counter += 1
    flag = True

