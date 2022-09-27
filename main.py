import openpyxl

dataFrame = openpyxl.load_workbook("/Users/maggiechen/Downloads/SpNdirDIA_QizhiSwab-23EvoToF24aug_quan.xlsx")

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

arr.sort()
print(arr)