from googleapiclient.discovery import build
from httplib2 import Http
from oauth2client import file, client, tools
import base64
from apiclient import errors
import os
import time
from google.cloud import pubsub_v1
from google.oauth2 import service_account
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import csv
import io
import cv2
import random
import math
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json
from google.cloud import automl_v1beta1 as automl

#AutoML
from google.cloud import automl_v1beta1
from google.cloud.automl_v1beta1.proto import service_pb2



SCOPES = 'https://www.googleapis.com/auth/gmail.readonly'
credentials = service_account.Credentials.from_service_account_file('visionAPI.json')


#Auto ML

project_id = 'sandbox-nguyenvietcuong'
model_NP_id = 'TCN2036296203375337149'
model_Obj_id= 'IOD1961638695115161600'
project_obj_id='1093967521155'

def get_prediction_NP(content, project_id, model_NP_id):
    prediction_client = automl_v1beta1.PredictionServiceClient()
    name = 'projects/{}/locations/us-central1/models/{}'.format(project_id, model_NP_id)
    payload = {'text_snippet': {'content': content, 'mime_type': 'text/plain' }}
    params = {}
    request = prediction_client.predict(name, payload, params)
    return request  # waits till request is returned

def get_prediction_Object_detection(file_path, project_obj_id, model_Obj_id):
    object_coordinate=[]
    with open(file_path, 'rb') as ff:
        content = ff.read()
    img=cv2.imread(file_path)
    prediction_client = automl_v1beta1.PredictionServiceClient()
    name = 'projects/{}/locations/us-central1/models/{}'.format(project_id, model_Obj_id)
    payload = {'image': {'image_bytes': content }}
    params = {}
    request = prediction_client.predict(name, payload, params)
    #print(request)
    for item in request.payload:
        label=item.display_name
        score=item.image_object_detection.score
        xmin=item.image_object_detection.bounding_box.normalized_vertices[0].x*img.shape[1]
        ymin=item.image_object_detection.bounding_box.normalized_vertices[0].y*img.shape[0]
        xmax=item.image_object_detection.bounding_box.normalized_vertices[1].x*img.shape[1]
        ymax=item.image_object_detection.bounding_box.normalized_vertices[1].y*img.shape[0]
        object_coordinate.append([label,xmin,ymin,xmax,ymax,score])
        #cv2.rectangle(img,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(0,255,0),1)
    return object_coordinate  # waits till request is returned


def opengspreadsheet(row,col,content):
    scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/spreadsheets',"https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("visionAPI.json", scope)
    client = gspread.authorize(creds)

    sheet = client.open("output").sheet1  # Open the spreadhseet

    #data = sheet.get_all_records()  # Get a list of all records

    #row = sheet.row_values(3)  # Get a specific row
    #col = sheet.col_values(3)  # Get a specific column
    #cell = sheet.cell(1,2).value  # Get the value of a specific cell

    #insertRow = ["hello", 5, "red", "blue"]
    #sheet.add_rows(insertRow, 4)  # Insert the list as a row at index 4

    sheet.update_cell(row,col, content)  # Update one cell

    numRows = sheet.row_count

                            
def detect_document(path):
    """Detects document features in an image."""
    from google.cloud import vision
    client = vision.ImageAnnotatorClient(credentials=credentials)

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)
    #nhan dien line
    imagedraw=cv2.imread(path)
    imagecheck=cv2.imread(path)
    breaks = vision.enums.TextAnnotation.DetectedBreak.BreakType
    paragraphs = []
    lines = []
    response = client.document_text_detection(image=image)
    boxlines=[]
    allRect=box_extraction(path)
    for item in allRect:
        xmin=item[0]
        ymin=item[1]
        xmax=item[2]
        ymax=item[3]
        cv2.rectangle(imagedraw,(xmin,ymin),(xmax,ymax),(255,255,0),2)
    boxrow=[]
    startCell=[]
    rowCell=[]
    paragraphCoordinate=[]
    
    object_coor=get_prediction_Object_detection(path,project_obj_id,model_Obj_id)
    
    for page in response.full_text_annotation.pages:
        # print("noi dung page")
        # print(page.blocks)
        for block in page.blocks:
            #print('\nBlock confidence: {}\n'.format(block.confidence))
            #print("----------------------")
            #print(block.bounding_box)
            xminb=block.bounding_box.vertices[0].x
            yminb=block.bounding_box.vertices[0].y
            xmaxb=block.bounding_box.vertices[2].x
            ymaxb=block.bounding_box.vertices[2].y
            #cv2.rectangle(imagedraw,(xminb,yminb),(xmaxb,ymaxb),(255,0,0),2)
            for paragraph in block.paragraphs:
                #print("paragraph-------------------------------------")
                #print(paragraph.bounding_box.vertices[0].x)
                xmin=paragraph.bounding_box.vertices[0].x
                ymin=paragraph.bounding_box.vertices[0].y
                xmax=paragraph.bounding_box.vertices[2].x
                ymax=paragraph.bounding_box.vertices[2].y
                
                #cv2.rectangle(imagedraw,(xmin,ymin),(xmax,ymax),(0,255,0),2)
                checkxmin=10000
                checkxmax=0
                para = ""
                line = ""
                box=[]
                #Dang xu ly lay toa do line da noi
                

                #print('Paragraph confidence: {}'.format(
                #    paragraph.confidence))
                for word in paragraph.words:
                    #print("word content")
                    #print(word)
                    xminword=word.bounding_box.vertices[0].x
                    yminword=word.bounding_box.vertices[0].y
                    xmaxword=word.bounding_box.vertices[2].x
                    ymaxword=word.bounding_box.vertices[2].y
                    if xminword<checkxmin:
                        checkxmin=xminword
                    if xmaxword>checkxmax:
                        checkxmax=xmaxword
                    cv2.rectangle(imagedraw,(xminword,yminword),(xmaxword,ymaxword),(0,255,255),1)
                    
                    #print("word------------")
                    #print(word.bounding_box.vertices[0].x)
                    # word_text = ''.join([
                        # symbol.text for symbol in word.symbols
                    # ])
                    
                    #Get Line   
                    for symbol in word.symbols:
                        #print("symbol box")
                        #print (symbol.bounding_box)
                        xminSymbol=symbol.bounding_box.vertices[0].x
                        yminSymbol=symbol.bounding_box.vertices[0].y
                        xmaxSymbol=symbol.bounding_box.vertices[2].x
                        ymaxSymbol=symbol.bounding_box.vertices[2].y
                        confidence=symbol.confidence
                        
                        if symbol.text=="〒":
                            paragraphCoordinate.append([xminb,yminb,xmaxb,ymaxb])
                            #print("block chua ky tu dia chi")
                            #print(confidence)
                        cv2.rectangle(imagedraw,(xminSymbol,yminSymbol),(xmaxSymbol,ymaxSymbol),(0,255,0),1)
                        line += symbol.text
                        if len(box)>0:
                            if xminSymbol < int(box[0][0]) :
                                box[0][0]=xminSymbol
                            if xmaxSymbol > int(box[0][2]) :
                                box[0][2]=xmaxSymbol
                            if yminSymbol < int(box[0][1]):
                                box[0][1]=yminSymbol
                            if ymaxSymbol > int(box[0][3]) :
                                box[0][3]=ymaxSymbol
                        else:
                            box.append([xminSymbol,yminSymbol,xmaxSymbol,ymaxSymbol,confidence])
                        if symbol.property.detected_break.type == breaks.SPACE:
                            #print(word)
                            #cv2.rectangle(imagedraw,(xminSymbol,yminSymbol),(xmaxSymbol,ymaxSymbol),(0,255,0),1)
                            line += symbol.text
                            if len(box)>0:
                                if xminSymbol < int(box[0][0]) :
                                    box[0][0]=xminSymbol
                                if xmaxSymbol > int(box[0][2]) :
                                    box[0][2]=xmaxSymbol
                                if yminSymbol < int(box[0][1]):
                                    box[0][1]=yminSymbol
                                if ymaxSymbol > int(box[0][3]) :
                                    box[0][3]=ymaxSymbol
                            else:
                                box.append([xminSymbol,yminSymbol,xmaxSymbol,ymaxSymbol,confidence])
                            # line += ' '
                            # lines.append([line,box[0][0],box[0][1],box[0][2],box[0][3]])
                            # para += line
                            # line = ''
                            # boxlines.append(box)
                            # box=[]
                            
                        if symbol.property.detected_break.type == breaks.EOL_SURE_SPACE:
                            line += ' '
                            lines.append([line,box[0][0],box[0][1],box[0][2],box[0][3],box[0][4]])
                            para += line
                            line = ''
                            boxlines.append(box)
                            box=[]
                        if symbol.property.detected_break.type == breaks.LINE_BREAK:
                            lines.append([line,box[0][0],box[0][1],box[0][2],box[0][3],box[0][4]])
                            para += line
                            line = ''
                            boxlines.append(box)
                            box=[]
                            
                    #print("lines gia tri")
                    #print(lines)
                paragraphs.append(para)
    
    
    print("done")
    #Check box of text start on the left
    customer=[]
    invoice_detail=[]
    pay_method=[]
    table=[]
    topgate=[]
    total=[]
    other=[]
    total_box=[]
    
    for item in object_coor:
        label1=item[0]
        xmin1=item[1]
        ymin1=item[2]
        xmax1=item[3]
        ymax1=item[4]
        allString=""
        if label1=="total_box":
            for text in lines:
                textvalue=text[0]
                xminLines=text[1]
                yminLines=text[2]
                xmaxLines=text[3]
                ymaxLines=text[4]
                xcenter=(xminLines+xmaxLines)/2
                ycenter=(yminLines+ymaxLines)/2
                
                if xcenter>xmin1 and xcenter<xmax1 and ycenter>ymin1 and ycenter<ymax1:
                    allString+=" " + textvalue
            total_box.append([allString])            
            
                     
    for allValue in lines:
        textvalue=allValue[0]
        xminLines=allValue[1]
        yminLines=allValue[2]
        xmaxLines=allValue[3]
        ymaxLines=allValue[4]
        # with open('result.txt', 'a', encoding='utf-8') as file:
            # file.write(textvalue)
            # file.write("----------")
        #cv2.rectangle(imagedraw,(xminLines,yminLines),(xmaxLines,ymaxLines),(0,255,0),1)
        xcenter=(xminLines+xmaxLines)/2
        ycenter=(yminLines+ymaxLines)/2
        flag=True
        for item in object_coor:
            label=item[0]
            xmin1=item[1]
            ymin1=item[2]
            xmax1=item[3]
            ymax1=item[4]
            if label=="customer":
                if xcenter>xmin1 and xcenter<xmax1 and ycenter>ymin1 and ycenter<ymax1 :
                    customer.append([textvalue,xminLines,yminLines,xmaxLines,ymaxLines])
                    flag=False
            if label=="invoice_detail":
                if xcenter>xmin1 and xcenter<xmax1 and ycenter>ymin1 and ycenter<ymax1 :
                    invoice_detail.append([textvalue,xminLines,yminLines,xmaxLines,ymaxLines])
                    flag=False
            if label=="table" :
                if xcenter>xmin1 and xcenter<xmax1 and ycenter>ymin1 and ycenter<ymax1  :
                    table.append([textvalue,xminLines,yminLines,xmaxLines,ymaxLines])
                    flag=False
            if label=="topgate" :
                if xcenter>xmin1 and xcenter<xmax1 and ycenter>ymin1 and ycenter<ymax1 :
                    topgate.append([textvalue,xminLines,yminLines,xmaxLines,ymaxLines])
                    flag=False
            if label=="total" :        
                if xcenter>xmin1 and xcenter<xmax1 and ycenter>ymin1 and ycenter<ymax1:
                    total.append([textvalue,xminLines,yminLines,xmaxLines,ymaxLines])
                    flag=False
            if label=="pay_method":        
                if xcenter>xmin1 and xcenter<xmax1 and ycenter>ymin1 and ycenter<ymax1 :
                    pay_method.append([textvalue,xminLines,yminLines,xmaxLines,ymaxLines])
                    flag=False
            if label == "total_box":
                if xcenter>xmin1 and xcenter<xmax1 and ycenter>ymin1 and ycenter<ymax1 :
                    flag=False
           
        if flag==True:
            other.append([textvalue,xminLines,yminLines,xmaxLines,ymaxLines])

    
    # TOPGATE BOX

    if len(topgate)!=0:
        textfirstbox=[]
        row=1
        column=1
        opengspreadsheet(row,column,"TOPGATE")
        for value in topgate:
            line=value[0]
            xmin=value[1]
            ymin=value[2]
            xmax=value[3]
            ymax=value[4]
            flag=True
            for item in topgate:
                xmin1=item[1]
                ymin1=item[2]
                xmax1=item[3]
                ymax1=item[4]
                if xmin1<xmin and xmax1<xmax and abs(ymin1-ymin)<10 and abs (ymax1-ymax)<10:
                    flag=False
            if flag==True:
                textfirstbox.append([xmin,ymin,xmax,ymax,line])    
        for value in  textfirstbox:
            row=row+1
            xmin2=value[0]
            ymin2=value[1]
            xmax2=value[2]
            ymax2=value[3]
            newString=""
            for var in topgate:
                line=var[0]
                xmin3=var[1]
                ymin3=var[2]
                xmax3=var[3]
                ymax3=var[4]
                
                if xmin3>=xmin2 and xmax3>=xmax2 and abs(ymin2-ymin3)<20 and abs(ymax2-ymax3)<20:
                    newString=newString+ " " + line
            opengspreadsheet(row,column,newString)
    
    # CUSTOMER BOX
    
    if len(customer)!=0:
        textfirstbox=[]
        
        row=1
        column=2
        opengspreadsheet(row,column,"CUSTOMER")
        for value in customer:
            line=value[0]
            xmin=value[1]
            ymin=value[2]
            xmax=value[3]
            ymax=value[4]
            flag=True
            for item in customer:
                xmin1=item[1]
                ymin1=item[2]
                xmax1=item[3]
                ymax1=item[4]
                if xmin1<xmin and xmax1<xmax and abs(ymin1-ymin)<10 and abs (ymax1-ymax)<10:
                    flag=False
            if flag==True:
                textfirstbox.append([xmin,ymin,xmax,ymax,line])    
        for value in  textfirstbox:
            row=row+1
            xmin2=value[0]
            ymin2=value[1]
            xmax2=value[2]
            ymax2=value[3]
            newString=""
            for var in customer:
                line=var[0]
                xmin3=var[1]
                ymin3=var[2]
                xmax3=var[3]
                ymax3=var[4]
                
                if xmin3>=xmin2 and xmax3>=xmax2 and abs(ymin2-ymin3)<20 and abs(ymax2-ymax3)<20:
                    newString=newString+ " " + line
            opengspreadsheet(row,column,newString)            
    
    #INVOICE DETAIL BOX
    
    if len(invoice_detail)!=0:
        textfirstbox=[]
        row=1
        column=3
        opengspreadsheet(row,column,"INVOICE DETAIL")
        for value in invoice_detail:
            line=value[0]
            xmin=value[1]
            ymin=value[2]
            xmax=value[3]
            ymax=value[4]
            flag=True
            for item in invoice_detail:
                xmin1=item[1]
                ymin1=item[2]
                xmax1=item[3]
                ymax1=item[4]
                if xmin1<xmin and xmax1<xmax and abs(ymin1-ymin)<10 and abs (ymax1-ymax)<10:
                    flag=False
            if flag==True:
                textfirstbox.append([xmin,ymin,xmax,ymax,line])    
        for value in  textfirstbox:
            row=row+1
            xmin2=value[0]
            ymin2=value[1]
            xmax2=value[2]
            ymax2=value[3]
            newString=""
            for var in invoice_detail:
                line=var[0]
                xmin3=var[1]
                ymin3=var[2]
                xmax3=var[3]
                ymax3=var[4]
                
                if xmin3>=xmin2 and xmax3>=xmax2 and abs(ymin2-ymin3)<20 and abs(ymax2-ymax3)<20:
                    newString=newString+ " " + line
            opengspreadsheet(row,column,newString)     
    
    #TABLE BOX

    if len(table)!=0:
        textfirstbox=[]
        row=1
        column=4
        opengspreadsheet(row,column,"TABLE DETAIL")
        for value in table:
            line=value[0]
            xmin=value[1]
            ymin=value[2]
            xmax=value[3]
            ymax=value[4]
            flag=True
            for item in table:
                xmin1=item[1]
                ymin1=item[2]
                xmax1=item[3]
                ymax1=item[4]
                if xmin1<xmin and xmax1<xmax and abs(ymin1-ymin)<10 and abs (ymax1-ymax)<10:
                    flag=False
            if flag==True:
                textfirstbox.append([xmin,ymin,xmax,ymax,line])    
        for value in  textfirstbox:
            row=row+1
            xmin2=value[0]
            ymin2=value[1]
            xmax2=value[2]
            ymax2=value[3]
            newString=""
            for var in table:
                line=var[0]
                xmin3=var[1]
                ymin3=var[2]
                xmax3=var[3]
                ymax3=var[4]
                
                if xmin3>=xmin2 and xmax3>=xmax2 and abs(ymin2-ymin3)<20 and abs(ymax2-ymax3)<20:
                    newString=newString+ " " + line
            opengspreadsheet(row,column,newString)  
    
    #TOTAL BOX
    
    if len(total)!=0 or len(total_box)!=0:
        
        textfirstbox=[]
        row=1
        column=5
        opengspreadsheet(row,column,"TOTAL BILL")
        for value in total:
            line=value[0]
            xmin=value[1]
            ymin=value[2]
            xmax=value[3]
            ymax=value[4]
            flag=True
            for item in total:
                xmin1=item[1]
                ymin1=item[2]
                xmax1=item[3]
                ymax1=item[4]
                if xmin1<xmin and xmax1<xmax and abs(ymin1-ymin)<10 and abs (ymax1-ymax)<10:
                    flag=False
            if flag==True:
                textfirstbox.append([xmin,ymin,xmax,ymax,line])    
        
        for stringData in total_box:
            row=row+1
            opengspreadsheet(row,column,stringData[0])
            
        for value in  textfirstbox:
            row=row+1
            xmin2=value[0]
            ymin2=value[1]
            xmax2=value[2]
            ymax2=value[3]
            newString=""
            
            for var in total:
                line=var[0]
                xmin3=var[1]
                ymin3=var[2]
                xmax3=var[3]
                ymax3=var[4]

                if xmin3>=xmin2 and xmax3>=xmax2 and abs(ymin2-ymin3)<20 and abs(ymax2-ymax3)<20:
                    newString=newString+ " " +line
            opengspreadsheet(row,column,newString)       

    #PAYMENT METHOD BOX

    if len(pay_method)!=0:
        textfirstbox=[]
        row=1
        column=6
        opengspreadsheet(row,column,"PAY METHOD")
        for value in pay_method:
            line=value[0]
            xmin=value[1]
            ymin=value[2]
            xmax=value[3]
            ymax=value[4]
            flag=True
            for item in pay_method:
                xmin1=item[1]
                ymin1=item[2]
                xmax1=item[3]
                ymax1=item[4]
                if xmin1<xmin and xmax1<xmax and abs(ymin1-ymin)<10 and abs (ymax1-ymax)<10:
                    flag=False
            if flag==True:
                textfirstbox.append([xmin,ymin,xmax,ymax,line])    
        print(textfirstbox)
        for value in  textfirstbox:
            row=row+1
            xmin2=value[0]
            ymin2=value[1]
            xmax2=value[2]
            ymax2=value[3]
            newString=""
            for var in pay_method:
                line=var[0]
                xmin3=var[1]
                ymin3=var[2]
                xmax3=var[3]
                ymax3=var[4]
                
                if xmin3>=xmin2 and xmax3>=xmax2 and abs(ymin2-ymin3)<20 and abs(ymax2-ymax3)<20:
                    newString=newString+ " " + line
            print(newString)
            opengspreadsheet(row,column,newString)  
    
    # OTHER TEXT
    if len(other)!=0:
        textfirstbox=[]
        row=1
        column=7
        opengspreadsheet(row,column,"OTHER CONTENT")
        for value in other:
            line=value[0]
            xmin=value[1]
            ymin=value[2]
            xmax=value[3]
            ymax=value[4]
            
            flag=True
            for item in other:
                xmin1=item[1]
                ymin1=item[2]
                xmax1=item[3]
                ymax1=item[4]
                if xmin1<xmin and xmax1<xmax and abs(ymin1-ymin)<20 and abs (ymax1-ymax)<20:
                    flag=False
            if flag==True:
                textfirstbox.append([xmin,ymin,xmax,ymax,line])
                #cv2.rectangle(imagedraw,(xmin,ymin),(xmax,ymax),(255,255,0),2)                
        #print(textfirstbox)
        for value in textfirstbox:
            row=row+1
            xmin2=value[0]
            ymin2=value[1]
            xmax2=value[2]
            ymax2=value[3]
            lines=value[4]
            newString=""
            for var in other:
                line=var[0]
                xmin3=var[1]
                ymin3=var[2]
                xmax3=var[3]
                ymax3=var[4]
                
                if xmin3>=xmin2 and xmax3>=xmax2 and abs(ymin2-ymin3)<20 and abs(ymax2-ymax3)<20:
                    newString=newString+ " " + line
            #print(newString)
            opengspreadsheet(row,column,newString)
    
    cv2.imwrite("result.jpg",imagedraw)


def getimage():
    # project_id = 
    # subscription_name =
    store_dir=os.getcwd()
    store = file.Storage('token.json')
    print(store)
    creds = store.get()
    print(type(creds))
    # subscriber = pubsub_v1.SubscriberClient()
    # subscription_path = subscriber.subscription_path(project_id, subscription_name)
    if not creds or creds.invalid:
        flow = client.flow_from_clientsecrets('credentials.json', SCOPES)
        creds = tools.run_flow(flow, store)
    service = build('gmail', 'v1', http=creds.authorize(Http()))
    print ("Service hien tai",service)
    result = service.users().messages().list(userId='me',labelIds = ['INBOX']).execute()
    #print(result)
    msg_id=result.get('messages',[])
    #print(msg_id)
    for value in msg_id:
        print (value["id"])
    
    try:
        message = service.users().messages().get(userId='me', id='16cd62ad4ca56a4b').execute()

        for part in message['payload']['parts']:
          if part['filename']:        
            attachment = service.users().messages().attachments().get(userId='me', messageId=message['id'], id=part['body']['attachmentId']).execute()
            file_data = base64.urlsafe_b64decode(attachment['data'].encode('UTF-8'))

            path = ''.join([store_dir, part['filename']])
            #print(path)
            f = open(path, 'wb')
            f.write(file_data)
            f.close()
    
    except errors.HttpError as error:
        print(f'An error occurred: {error}')
    
    return path
    # Call the Gmail API to fetch INBOX
    # results = service.users().messages().list(userId='me',labelIds = ['INBOX']).execute()
    # messages = results.get('messages', [])
    

    # if not messages:
        # print ("No messages found.")
    # else:
        # print ("Message snippets:")
        # for message in messages:
            # msg = service.users().messages().get(userId='me', id=message['id']).execute()
            # if msg['filename']:
            
                # print (msg)



def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

def box_extraction(img_for_box_extraction_path):

    img = cv2.imread(img_for_box_extraction_path, 0)  # Read the image
    #print(img)
    thresh, img_bin = cv2.threshold(img, 230, 255,cv2.THRESH_BINARY )  # Thresholding the image
    #thresh, img_bin = cv2.threshold(img, 20, 255,cv2.THRESH_TOZERO)
    img_bin = 255-img_bin  # Invert the image

    cv2.imwrite("Image_bin.jpg",img_bin)
   
    # Defining a kernel length
    kernel_length = np.array(img).shape[1]//80

    # A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    # A kernel of (3 X 3) ones.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Morphological operation to detect verticle lines from an image
    img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=4)
    #cv2.imwrite("verticle_lines.jpg",verticle_lines_img)

    # Morphological operation to detect horizontal lines from an image
    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=4)
    #cv2.imwrite("horizontal_lines.jpg",horizontal_lines_img)

    # Weighting parameters, this will decide the quantity of an image to be added to make a new image.
    alpha = 0.5
    beta = 1.0 - alpha
    # This function helps to add two image with specific weight parameter to get a third image as summation of two image.
    img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
    
    (thresh, img_final_bin) = cv2.threshold(img_final_bin, 120, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # For Debugging
    # Enable this line to see verticle and horizontal lines in the image which is used to find boxes
    #cv2.imwrite("img_final_bin.jpg",img_final_bin)
    # Find contours for image, which will detect all the boxes
    contours, hierarchy = cv2.findContours(img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Sort all the contours by top to bottom.
    (contours, boundingBoxes) = sort_contours(contours, method="top-to-bottom")

    idx = 0
    allRect=[]
    allRectFilter=[]
    for c in contours:
        # Returns the location and width,height for every contour
        x, y, w, h = cv2.boundingRect(c)
       
        
        # If the box height is greater then 20, widht is >80, then only save it as a box in "cropped/" folder.
        if (w > 30 and h > 10):
            allRect.append([x,y,x+w,y+h])
            idx += 1
            new_img = img[y:y+h, x:x+w]
            #cv2.imwrite(cropped_dir_path+str(idx) + '.png', new_img)
    
    for value in allRect:
        xmin=value[0]
        ymin=value[1]
        xmax=value[2]
        ymax=value[3]
        flag=True
        for box in allRect:
            xmin1=box[0]
            ymin1=box[1]
            xmax1=box[2]
            ymax1=box[3]
            if xmin1>xmin and ymin1>ymin and xmax1<xmax and ymax1<ymax:
                ymin=ymin1
                flag=False
        if flag==True:
            allRectFilter.append([xmin,ymin,xmax,ymax])
    
   # print(allRectFilter)
    return allRectFilter

if __name__ == '__main__':
    #pathimage=getimage()
    #path="C:\\Users\\Nguyen Viet CUONG\\Desktop\\labGCP\\readbill\\image\\13.jpg"
    #getimage()
    
    #arrText=detect_text("C:\\Users\\Nguyen Viet CUONG\\Desktop\\labGCP\\readbill\\1.jpg")
    detect_document("C:\\Users\\Nguyen Viet CUONG\\Desktop\\labGCP\\readbill\\image\\page124.jpg")
    #async_detect_document("gs://sandbox-nguyenvietcuong/data_files/1.pdf","gs://sandbox-nguyenvietcuong/data_files")
