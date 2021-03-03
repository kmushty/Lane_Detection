
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import glob
import math
import yaml
import copy

# img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

source_points = []
def draw_circle(event,x,y,flags,param):
    global mouseX,mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        mouseX,mouseY = x,y
        print(mouseX, mouseY)
        source_points.append([mouseX,mouseY])

path=os.getcwd()
##refpath=os.path.join(path, 'PS2_Videos/data_1/data')
vidpath=os.path.join(path, 'PS2_Videos/data_1/data')
vidcap = cv2.VideoCapture(vidpath + '/project.avi')

print(vidcap.isOpened())

while (vidcap.isOpened()):
    ret, img = vidcap.read()
    if ret == True:
        ##img=cv2.imread(refpath + '/0000000000.png')

        ###############################################
        #start Undistort and de-noise

        #Read camera matrix and distortion values
        CamMatrix = []
        camValPath = os.path.join(path, 'PS2_Videos/data_1')
        with open(camValPath + '/camera_params.yaml') as CamFile:
            CamDoc = yaml.load(CamFile)
            K = CamDoc["K"]
            D = CamDoc["D"]

        count = 0
        CamDocK = [float(x) for x in K.split()]
        DCoeff = [float(d) for d in D.split()]
        ##CamMatrix = [[0 for x in range(3)] for y in range (3)]
        CamMatrix=np.zeros((3,3))
        for k in range(0,3):
            for i in range(0,3):
                CamMatrix[k][i]=CamDocK[count]
                count+=1

        DistCoeff = np.zeros((1,5))

        for d in range(0,5):
            DistCoeff[0][d] = DCoeff[d]

        #Perform undistort function
        ImgDist = cv2.undistort(img,CamMatrix,DistCoeff)
        cv2.imshow("Image after Distortion", ImgDist)

        ##ImgGray = cv2.cvtColor(ImgDist, cv2.COLOR_BGR2GRAY)
        ##ImgGauss = cv2.GaussianBlur(ImgGray, (5,5), cv2.BORDER_CONSTANT)
        ##ImgBlur = cv2.medianBlur(ImgGray, 5)
        ##ImgFilter = cv2.bilateralFilter(ImgGray, 9, 75, 75)

        ImgDeNoise = cv2.fastNlMeansDenoisingColored(ImgDist,None,8,10,7,21)

##        cv2.imshow("Image De-noise", ImgDeNoise)

        #end Undistort and de-noise
        #############################################
        #start Color conversion and image prep

        ImgGray = cv2.cvtColor(ImgDeNoise, cv2.COLOR_RGB2GRAY)

        bright = 210
        h, w = ImgGray.shape

        i = 0
        k = 0
        ImgDark = np.zeros((h,w))
        for i in range(0,h):
            for k in range(0,w):
                if ImgGray[i][k]<=bright:
                    ImgDark[i][k]=0
                else:
                    ImgDark[i][k]=ImgGray[i][k]-bright

        ImgGauss = cv2.GaussianBlur(ImgDark, (5,5), cv2.BORDER_CONSTANT)

##        cv2.imshow("Image after Gaussian Blur", ImgGauss)

        ImgHLS = cv2.cvtColor(ImgDeNoise, cv2.COLOR_RGB2HLS)

        #end Color conversion and image prep
        ############################################
        #start Canny Filter

        # converting Gray to HSV 
        hsv = cv2.cvtColor(ImgDeNoise, cv2.COLOR_RGB2HSV) 
              
        # define range of red color in HSV 
        lower_red = np.array([30,150,50]) 
        upper_red = np.array([255,255,180]) 
              
        # create a red HSV colour boundary and  
        # threshold HSV image 
        mask = cv2.inRange(hsv, lower_red, upper_red) 
          
        # Bitwise-AND mask and original image 
        res = cv2.bitwise_and(img,img, mask= mask) 
          
        # finds edges in the input image image and 
        # marks them in the output mapedges

        ##ImgCanny = cv2.Canny(hsv,100,230)

        ImgSobelX = cv2.Sobel(ImgGauss,cv2.CV_8U,1,0,ksize=5)
        ImgSobelY = cv2.Sobel(ImgGauss,cv2.CV_8U,0,1,ksize=5)
        ImgSobel = ImgSobelX + ImgSobelY

        ##cv2.imshow("IMG Canny",ImgCanny)
        ##cv2.waitKey(0)

        #end Canny Filter
        ####################################################3
        #start ROI

        ImgLanes = copy.deepcopy(ImgSobel)
        ImgLanes[0:280,:]=0

##        cv2.imshow("Image after Sobel Function", ImgLanes)

        ##cv2.imshow("IMG Lanes",ImgLanes)
        ##cv2.waitKey(0)

        #end ROI
        ###############
        #start Homography and Warp

        ##width=img.shape[0]
        ##height=img.shape[1]
        ##cv2.namedWindow('Finding the Four Points')
        ##cv2.setMouseCallback('Finding the Four Points', draw_circle)

        # Uncomment only if you want to select different points for estimating homoography
        ##while True:
        ##    print('Double Click to store new points; Otherwise PRESS ESC')
        ##    print('Select point in clockwise manner starting from top-left')
        ##    # cv2.imshow('Finding the Four Points', img)
        ##    # cv2.waitKey(0)
        ##
        ##    cv2.destroyAllWindows()
        ##
        ##    break
        src=np.array([[180,400],[450,300],[740,300],[800,400]])
        dst = np.array([[100,200], [1000 ,200], [1000 ,600], [100 ,600]],dtype='f')

        H,status=cv2.findHomography(src,dst)
        ImgWarp=cv2.warpPerspective(ImgLanes,H,(800,1000))
        ImgRotate = cv2.rotate(ImgWarp, cv2.ROTATE_90_COUNTERCLOCKWISE)

        #end Homography and Warp
        ###############################################
        #start Lane Detection
        ###### Hough Lines ######

        ##ImgHough = cv2.HoughLines(ImgLanes,1,np.pi/180,200)
        ##print(ImgHough)
        ##for x1,y1,x2,y2 in ImgHough[0]:
        ##    cv2.line(ImgLanes,(x1,y1),(x2,y2),(0,255,0),2)
        ##cv2.imshow("Hough Lines", ImgLanes)
        ##cv2.waitKey(0)
        ##cv2.destroyAllWindows()

        ImgHough = cv2.HoughLines(ImgLanes,1,np.pi/180,200)

        ##ImgHough = cv2.HoughLinesP(ImgLanes,1,np.pi/180,50,None,50,225)

        LeftSide = []
        RightSide = []
        for line in ImgHough:
            rho, theta = line[0]
            if rho < np.pi/2:
                LeftSide.append(line)
            elif rho > np.pi/2:
                RightSide.append(line)
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            cv2.line(ImgLanes,(x1,y1),(x2,y2),(50,150,50),2)

##        cv2.imshow("Image with Hough Lines", ImgLanes)
##        cv2.waitKey(0)
##        cv2.destroyAllWindows()

        LeftSide = np.asarray(LeftSide)
        RightSide = np.asarray(RightSide)

##        LSobj,LSrow,LSitems = LeftSide.shape
##        RSobj,RSrow,RSitems = RightSide.shape

        if LeftSide.size!=0:
            PreviousLS = copy.deepcopy(LeftSide)
            LSobj,LSrow,LSitems = LeftSide.shape
        if RightSide.size!=0:
            PreviousRS = copy.deepcopy(RightSide)
            RSobj,RSrow,RSitems = RightSide.shape

 ### In case not enough lines are detected for one side,
 ### Use previously detected line
        if LeftSide.size==0:
            LSobj,LSrow,LSitems = PreviousLS.shape
            LeftSide=copy.deepcopy(PreviousLS)
        if RightSide.size==0:
            RSobj,RSrow,RSitems = PreviousRS.shape
            RightSide=copy.deepcopy(PreviousRS)

        j=0
        k=0
        LSrhoMean=0
        LSthetMean=0
        RSrhoMean=0
        RSthetMean=0
        for j in range(0,LSobj):
            LSrhoMean += LeftSide[j][0][0]
            LSthetMean += LeftSide[j][0][1]

        for k in range(0,RSobj):
            RSrhoMean += RightSide[k][0][0]
            RSthetMean += RightSide[k][0][1]

        LSrhoAve = LSrhoMean/LSobj
        LSthetAve = LSthetMean/LSobj
        RSrhoAve = RSrhoMean/RSobj
        RSthetAve = RSthetMean/RSobj

        LaneAve = np.array([[LSrhoAve,LSthetAve],[RSrhoAve,RSthetAve]])
        LaneLines=[]
        LanePoints=[]
        for AveLine in LaneAve:
            rho, theta = AveLine

            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            slope = (y2-y1)/float((x2-x1))
            intercept = y1 - slope * x1
            
            LaneLines.append([slope, intercept])

        LaneLines = np.asarray(LaneLines)

        LaneTrace = []
        for line in LaneLines:
            slope = line[0]
            intercept = line[1]
            y1 = h
            y2 = h-200

            x1 = int((y1 - intercept)/(slope))
            x2 = int((y2 - intercept)/(slope))

            LaneTrace.append([x1,y1])
            LaneTrace.append([x2,y2])

        #end Lane detection
        ######################################
        #start Lane tracing

        LaneTrace.append(LaneTrace.pop(2))
        LaneTrace=np.asarray(LaneTrace)

        cv2.polylines(img,np.int32([LaneTrace]),True,(255,0,0),3)

        overlay=img.copy()
        cv2.fillPoly(overlay,np.int32([LaneTrace]),(255,0,0))
        Tr = 0.4 #Transparency factor

        ImgLaneDisp = cv2.addWeighted(overlay,Tr,img,1-Tr,0)

        #end Lane tracing
        #######################################
        #start Direction and position prediction

        #Predicted Lane Direction
        LaneDirec = (LaneLines[0][0]+LaneLines[1][0])/2
        print("Predicted Road Bend:")
        if LaneDirec>0.05:
            print"Road is turning Left at a predicted angle of ",round(LaneDirec*(180/np.pi),2)," Degrees ahead"
        elif LaneDirec<0.05:
            print"Road is turning Right at a predicted angle of ",round(LaneDirec*(-180/np.pi),2)," Degrees ahead"
        else:
            print("Road is straight")

        y1 = h-200
        y2 = h
        x2 = (LaneTrace[0][0]+LaneTrace[3][0])/2
        x1 = (LaneTrace[1][0]+LaneTrace[2][0])/2

        HeadPoints = np.array([[x1,y1],[x2,y2]])

        cv2.polylines(ImgLaneDisp,np.int32([HeadPoints]),True,(0,255,0),2)

        VehCenter = w/2
        print("\n")
        print("Lane Position:")
        LanePos=VehCenter-x2
        if LanePos>=10:
            print("Vehicle is to the Right of lane center")
        elif LanePos<=-10:
            print("Vehicle is to the Left of lane center")
        else:
            print("Vehicle is at lane center")
        print("\n")

        #end Direction and position prediction
        #######################################
        #Display original image with Lane Detection

        cv2.destroyAllWindows()
        cv2.imshow("Lanes Detected", ImgLaneDisp)
        cv2.waitKey(3000)
        ##cv2.imshow("Turn Prediction",ImgLanes)
##        cv2.waitKey()
##        cv2.destroyAllWindows()

    else:
        break

vidcap.release()
cv2.destroyAllWindows()


