from PIL import Image, ImageDraw
import os
import json


path = "/home/rainer/Code/ocr-train/data/iam/testSet"

def check(ranges,ym,yM):
    for r in ranges:
        # caso in cui una word è più grande di un range
        if ym <= r[0] and r[1] <= yM:
            return (1,r)
        # caso in cui una word è più piccola di un range
        if r[0] <= ym and yM <= r[1]:
            return (2,r)
        # caso in cui la parte superiore di una word è sovrapposta a un range
        if r[0] <= ym and r[1] < yM:
            if int((ym + yM)/2) < r[1]:
                return (3,r)
        # caso in cui la parte inferiore di una word è sovrapposta a un range
        if yM <= r[1] and ym < r[0]:
            if int((ym + yM)/2) > r[0]:
                return (4,r)
    return None

def create_yranges(pred_lst):
    ranges = []
    for word in pred_lst:
        coord = word[1]
        ym = coord[1]
        yM = coord[3]

        result = check(ranges,ym,yM)
        if result == None:
            ranges.append((ym,yM))
        else:
            if result[0] == 1:
                ranges.remove(result[1])
                ranges.append((ym,yM))
            if result[0] == 3:
                ranges.remove(result[1])
                ranges.append((result[1][0],yM))
            if result[0] == 4:
                ranges.remove(result[1])
                ranges.append((ym,result[1][1]))
    return sorted(ranges,key=lambda x: x[0])

def orderByY(ranges,pred_lst):
    rows = [[] for _ in range(0,len(ranges))]
    for pred in pred_lst:
        coord = pred[1]
        yAvg = int((coord[1] + coord[3])/2)
        for i,r in enumerate(ranges):
            if r[0] < yAvg and yAvg < r[1]:
                rows[i].append((pred[0],coord[0]))
    return rows


# order by x the 
def orderByX(ySorted):
    xSorted = []
    for row in ySorted:
        xSorted.append(sorted(row,key=lambda x: x[1]))
    return xSorted

def createPredictionByLine(pred_lst):
    ySorted = orderByY(create_yranges(pred_lst),pred_lst)
    xSorted = orderByX(ySorted)
    
    # getting only the words inside xSorted
    pred_text = []
    for element in xSorted:
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
        fullPagePrediction = createPredictionByLine(pred_lst)
        jsonResult[image] = fullPagePrediction
    with open("/home/rainer/Code/ocr-benchmark-result/results/iam/doctr/result.json","w") as f:
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
        fullPagePrediction = createPredictionByLine(pred_lst)
        jsonResult[image] = fullPagePrediction
    with open("/home/rainer/Code/ocr-benchmark-result/results/iam/easyocr/result.json","w") as f:
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
        fullPagePrediction = createPredictionByLine(pred_lst)
        jsonResult[image] = fullPagePrediction
    with open("/home/rainer/Code/ocr-benchmark-result/results/iam/mmocr/result.json","w") as f:
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
        fullPagePrediction = createPredictionByLine(pred_lst)
        jsonResult[image] = fullPagePrediction
    with open("/home/rainer/Code/ocr-benchmark-result/results/iam/paddleocr/result.json","w") as f:
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

        fullPagePrediction = createPredictionByLine(pred_lst)
        jsonResult[image] = fullPagePrediction
    with open("/home/rainer/Code/ocr-benchmark-result/results/iam/tesseractocr/result.json","w") as f:
        json.dump(jsonResult,f,indent=4)
        


#print(doctr_draw())
#print(easyocr_draw())
#print(mmocr_draw())
#print(paddleocr_draw())
print(tesseract_draw())