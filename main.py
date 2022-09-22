import openpyxl

dataFrame = openpyxl.load_workbook("/Users/maggiechen/Downloads/SpNdirDIA_QizhiSwab-23EvoToF24aug_quan.xlsx")

dataFrameReader = dataFrame.active

for row in range(0, dataFrameReader.max_row):
    for col in dataFrameReader.iter_cols(3,3):
        print(col[row].value)

