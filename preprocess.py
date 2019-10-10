import cv2
import numpy as np


img = cv2.imread('8.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,100,200,apertureSize = 3)
minLineLength = 5
maxLineGap = 80
lines = cv2.HoughLinesP(edges,1,np.pi/180,15,minLineLength,maxLineGap)
for x in range(0, len(lines)):
    for x1,y1,x2,y2 in lines[x]:
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),1)



# find contours and get the external one
img=cv2.bitwise_not(img)
kernel = np.ones((5, 25), np.uint8)
img = cv2.dilate(img, kernel, iterations=1)
img = cv2.erode(img, kernel, iterations=1)

imgarr=np.array(img)

#img=cv2.bitwise_not(img)
# threshold image
ret, threshed_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                100, 255, cv2.THRESH_BINARY)
contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#image, contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE,
#                cv2.CHAIN_APPROX_SIMPLE)

# with each contour, draw boundingRect in green
# a minAreaRect in red and
# a minEnclosingCircle in blue
arrRect=[]
for c in contours:
    # get the bounding rect
    x, y, w, h = cv2.boundingRect(c)
    # draw a green rectangle to visualize the bounding rect
    #cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
    dientich=w*h
    if dientich<100000:
        arrRect.append([x,y,x+w,y+h])
    # get the min area rect
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    # convert all coordinates floating point values to int
    box = np.int0(box)
    # draw a red 'nghien' rectangle
    #cv2.drawContours(img, [box], 0, (0, 0, 255))

rectin=[]   
for value in arrRect:
    xmin=value[0]
    ymin=value[1]
    xmax=value[2]
    ymax=value[3]    
    for value2 in arrRect:
        xmin2=value2[0]
        ymin2=value2[1]
        xmax2=value2[2]
        ymax2=value2[3]
        if xmin>xmin2 and xmax<xmax2 and ymin>ymin2 and ymax<ymax2:
            rectin.append([xmin,ymin,xmax,ymax])
        elif xmin>xmin2 and xmax<=xmax2 and ymin>ymin2 and ymax<=ymax2:
            rectin.append([xmin,ymin,xmax,ymax])
        elif xmin>=xmin2 and xmax<xmax2 and ymin>=ymin2 and ymax<ymax2:
            rectin.append([xmin,ymin,xmax,ymax])
        elif xmin>xmin2 and xmax<=xmax2 and ymin>=ymin2 and ymax<ymax2:
            rectin.append([xmin,ymin,xmax,ymax])            
            
for value in arrRect:
    xmin=value[0]
    ymin=value[1]
    xmax=value[2]
    ymax=value[3]
    flag=True
    for value2 in rectin:
        xmin2=value2[0]
        ymin2=value2[1]
        xmax2=value2[2]
        ymax2=value2[3]
        if xmin==xmin2 and ymin==ymin2 and xmax ==xmax2 and ymax==ymax2:
            flag=False
    if flag==True:
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)

print ("arr rect",len(arrRect))
print("rect in",len(rectin))
#cv2.drawContours(img, contours, -1, (255, 255, 0), 1)

cv2.imwrite("result.png",img)


cv2.destroyAllWindows()