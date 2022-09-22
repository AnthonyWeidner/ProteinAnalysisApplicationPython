import openpyxl

dataFrame = openpyxl.load_workbook("/Users/maggiechen/Downloads/SpNdirDIA_QizhiSwab-23EvoToF24aug_quan.xlsx")

dataFrameReader = dataFrame.active
arr = []


for row in dataFrameReader.iter_rows(0, dataFrameReader.max_row):
    arr.append(row[4].value)

arr.sort()
print(arr)