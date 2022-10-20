import openpyxl
import tkinter.filedialog

root = tkinter.Tk()
root.withdraw
fileName1 = tkinter.filedialog.askopenfilename()

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

