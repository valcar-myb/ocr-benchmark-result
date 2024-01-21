from PIL import Image, ImageDraw
import os
import json
import time
import matplotlib.pyplot as plt


path = "/home/rainer/Code/ocr-train/data/iam/testSet"
number = 0

def check(rows_range,box):
    result = (0,box)
    for row in rows_range:
        # caso in cui box corrente è più grande di un range
        if box[0] < row[0] and row[1] < box[1]:
            result = (1,row)

        # caso in cui parte superiore del box è sovrapposto al range
        if row[0] <= box[0] and row[1] < box[1]:
            if row[1] - box[0] > int((box[1] - box[0])/2):
                result = (2,row)

        # caso in cui parte inferiore del box è sovrapposto al range
        if box[0] < row[0] and box[1] <= row[1]:
            if box[1] - row[0] > int((box[1] - box[0])/2):
                result = (3,row)

        # caso in cui box corrente è più piccolo del range
        if row[0] <= box[0] and box[1] <= row[1]:
            result = (4,row)
    return result

def prediction_rows(pred_lst):
    rows_range = []
    # pred -> (str * list)
    for pred in pred_lst:
        box = pred[1]
        ym = box[1]
        yM = box[3]
        check_result = check(rows_range,(ym,yM))

        id_case = check_result[0]
        result = check_result[1]

        if id_case == 0:
            rows_range.append(result)
        elif id_case == 1:
            rows_range.remove(result)
            rows_range.append((ym,yM))
        elif id_case == 2:
            rows_range.remove(result)
            rows_range.append((result[0],yM))
        elif id_case == 3:
            rows_range.remove(result)
            rows_range.append((ym,result[1]))
        elif id_case == 4:
            None
    # da qui si ottengono la maggior parte dei range
    yRange_sorted = sorted(rows_range,key = lambda x:x[0])
    # adesso si eliminano gli intervalli totalmente compresi
    ranges_to_delete = set()
    for i,x in enumerate(yRange_sorted):
        for j,y in enumerate(yRange_sorted):
            if i != j:
                if y[0] <= x[0] and x[1] <= y[1]:
                    ranges_to_delete.add(x)
    for rtd in ranges_to_delete:
        yRange_sorted.remove(rtd)

    return yRange_sorted

def sortByYAxis(yRange_sorted,pred_lst):
    lines = [[] for _ in range(0,len(yRange_sorted))]
    for pred in pred_lst:
        coord = pred[1]
        yAvg = int((coord[1] + coord[3])/2)
        for i,yr in enumerate(yRange_sorted):
            if yAvg in range(yr[0],yr[1]):
                lines[i].append((pred[0],coord[0]))
    return lines

def sortByXAxis(sortedByY):
    xSorted = []
    for row in sortedByY:
        xSorted.append(sorted(row,key=lambda x: x[1]))
    return xSorted

def orderPredictionByLine(pred_lst):
    yRange_sorted = prediction_rows(pred_lst)
    sortedByY = sortByYAxis(yRange_sorted,pred_lst)
    sortedByX = sortByXAxis(sortedByY)
    pred_text = []
    for element in sortedByX:
        pred_text.append([x[0] for x in element])

    # joining all the words in the respective row
    pred_line = []
    for element in pred_text:
        pred_line.append(' '.join(element))
    return pred_line

def doctr_draw(): 
    test_folder = os.listdir(path)
    test_folder.sort()
    jsonResult = dict()
    for image in test_folder:
        #print(image)
        img = Image.open(f"{path}/{image}")
        result = json.load(open(f"/home/rainer/Code/ocr-benchmark-result/results/partial/iam/doctr/{image[:-4]}.json"))

        draw = ImageDraw.Draw(img)
        pred_lst = []

        for page in result["pages"]:
            for block in page["blocks"]:
                for line in block["lines"]:
                    for word in line["words"]:
                        coord = [int(word["geometry"][0][0] * img.width),
                                 int(word["geometry"][0][1] * img.height),
                                 int(word["geometry"][1][0] * img.width),
                                 int(word["geometry"][1][1] * img.height)]
                        pred_lst.append((word["value"],coord))
                        #draw.rectangle(coord,outline="black")
                        #draw.text((coord[0],coord[1]),text=word["value"])
        #img.save(f"prova/{image}")
        jsonResult[image] = orderPredictionByLine(pred_lst)
    with open("results/iam/doctr/result.json","w") as f:
        json.dump(jsonResult,f,indent=4)
    
def easyocr_draw():
    test_folder = os.listdir(path)
    test_folder.sort()
    jsonResult = dict()
    for image in test_folder:
        img = Image.open(f"{path}/{image}")
        result = json.load(open(f"/home/rainer/Code/ocr-benchmark-result/results/partial/iam/easyocr/{image[:-4]}.json"))

        pred_lst = []
        draw = ImageDraw.Draw(img)

        for word in result["prediction"]:
            xm,ym,xM,yM = img.width,img.height,0,0
            for coord in word["boxes"]:
                xm = min(coord[0],xm)
                ym = min(coord[1],ym)
                xM = max(coord[0],xM)
                yM = max(coord[1],yM)
            pred_lst.append((word["text"],[int(xm),int(ym),int(xM),int(yM)]))
        jsonResult[image] = orderPredictionByLine(pred_lst)
    with open("results/iam/easyocr/result.json","w") as f:
        json.dump(jsonResult,f,indent=4)

def mmocr_draw():
    test_folder = os.listdir(path)
    test_folder.sort()
    jsonResult = dict()
    for image in test_folder:
        img = Image.open(f"{path}/{image}")
        result = json.load(open(f"/home/rainer/Code/ocr-benchmark-result/results/partial/iam/mmocr/{image[:-4]}.json"))

        prediction = result["predictions"][0]
        draw = ImageDraw.Draw(img)
        pred_lst = []
        
        for i,text in enumerate(prediction["rec_texts"]):
            xm,ym,xM,yM = img.width,img.height,0,0
            polygons = prediction["det_polygons"][i]
            for index in range(0,len(polygons),2):
                xm = min(polygons[index],xm)
                ym = min(polygons[index + 1],ym)
                xM = max(polygons[index],xM)
                yM = max(polygons[index + 1],yM)
                
            pred_lst.append((text,[int(xm),int(ym),int(xM),int(yM)]))
            #draw.rectangle((xm,ym,xM,yM),outline="black")
            #draw.text((xm,ym),text=text)
        #img.save(f"pictures/iam/{image}")
        jsonResult[image] = orderPredictionByLine(pred_lst)
    with open("results/iam/mmocr/result.json","w") as f:
        json.dump(jsonResult,f,indent=4)
    
def paddleocr_draw():
    test_folder = os.listdir(path)
    test_folder.sort()
    jsonResult = dict()
    for image in test_folder:
        img = Image.open(f"{path}/{image}")
        result = json.load(open(f"/home/rainer/Code/ocr-benchmark-result/results/partial/iam/paddleocr/{image[:-4]}.json"))

        prediction = result["prediction"][0]
        draw = ImageDraw.Draw(img)
        pred_lst = []
        
        for word in result["prediction"]:
            xm,ym,xM,yM = img.width,img.height,0,0
            for coord in word["box"]:
                xm = min(coord[0],xm)
                ym = min(coord[1],ym)
                xM = max(coord[0],xM)
                yM = max(coord[1],yM)
            pred_lst.append((word["text"],[int(xm),int(ym),int(xM),int(yM)]))
            #draw.rectangle((xm,ym,xM,yM),outline="black")
        jsonResult[image] = orderPredictionByLine(pred_lst)
    with open("results/iam/paddleocr/result.json","w") as f:
        json.dump(jsonResult,f,indent=4)
    
def tesseract_draw():
    test_folder = os.listdir(path)
    test_folder.sort()
    jsonResult = dict()
    for image in test_folder:
        img = Image.open(f"{path}/{image}")
        result = json.load(open(f"/home/rainer/Code/ocr-benchmark-result/results/partial/iam/tesseractocr/{image[:-4]}.json"))

        pred_lst = []

        #draw = ImageDraw.Draw(img)

        for i,text in enumerate(result["text"]):
            if result["conf"][i] != -1:
                coord = (
                    int(result["left"][i]),int(result["top"][i]),
                    int(result["left"][i]) + int(result["width"][i]),int(result["top"][i]) + int(result["height"][i])
                )
                pred_lst.append((text,coord))
        jsonResult[image] = orderPredictionByLine(pred_lst)
    with open("results/iam/tesseractocr/result.json","w") as f:
        json.dump(jsonResult,f,indent=4)



print(doctr_draw())
print(easyocr_draw())
print(mmocr_draw())
print(paddleocr_draw())
print(tesseract_draw())