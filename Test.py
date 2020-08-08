from Backbone import ResNet
from YoloV1 import YoloDetection
from PIL import Image
from abc import ABC
import torch.nn as nn
import torch
import numpy as np
import torchvision as tv

class YoloTrans(nn.Module, ABC):

    def __init__(self, BoundingBoxes: int, num_classes: int, SGrid: int, imageSize = 448, device = "cpu"):
        super().__init__()
        self.B = BoundingBoxes
        self.S = SGrid
        self.nc = num_classes
        self.imageSize = imageSize
        self.oneGridDis = imageSize // SGrid
        self.leftTopPositions = np.zeros(shape=[SGrid, SGrid, 2])
        for i in range(SGrid):
            for j in range(SGrid):
                self.leftTopPositions[i,j,:] = imageSize // SGrid * j , imageSize // SGrid * i ## x, y
        self.leftTopPositions = torch.from_numpy(self.leftTopPositions).float().to(device)
        self.device = device

    def TransformPredictedBoxes(self, preBoxes):
        """
        Transform yolo coordinate to left, top, right, bottom format.
        :param preBoxes: [N, S, S, BOXES * 4], (x, y, w, h), (x, y, w, h)
        :return: oriBoxes [N, S * S * Boxes, 4] (x1, y1, x2, y2)
        """
        with torch.no_grad():

            ### [N, S, S, B, 4]
            expandDim = torch.reshape(preBoxes, [-1, self.S, self.S, self.B, 4])
            ### [N, S, S, B, 2]
            xy, wh = torch.split(expandDim, [2, 2], dim=-1)
            currentLTP = torch.unsqueeze(self.leftTopPositions, dim=-2).unsqueeze(dim=0)  ## [1, S, S, 1, 2]
            xy = xy * self.oneGridDis
            # print(xy.device)
            # print(currentLTP.device)
            centerXY = xy + currentLTP
            ### [N, S, S, B, 2]
            oriWH = wh * self.imageSize
            centerXY = torch.reshape(centerXY, [-1,self.S * self.S * self.B, 2])
            oriWH = torch.reshape(oriWH, [-1,self.S * self.S * self.B, 2])
            x = centerXY[:,:,0]
            y = centerXY[:,:,1]
            w = oriWH[:,:,0]
            h = oriWH[:,:,1]
            x1 = torch.clamp(x - w / 2.,min=0, max=self.imageSize - 1)
            y1 = torch.clamp(y - h / 2.,min=0, max=self.imageSize - 1)
            x2 = torch.clamp(x + w / 2.,min=0, max=self.imageSize - 1)
            y2 = torch.clamp(y + h / 2.,min=0, max=self.imageSize - 1)
            return torch.stack([x1, y1, x2, y2],dim=-1) #, centerXY

import cv2
def drawBox(boxes, image):
    """
    :param boxes: np array, [N,4], (x1, y1, x2, y2)
    :param image: np array
    :return:
    """
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    numBox = boxes.shape[0]
    for j in range(0, numBox):
        # changed color and width to make it visible
        cv2.rectangle(image, (boxes[j,0], boxes[j,1]), (boxes[j,2], boxes[j,3]), (0, 0, 0), 2)
    cv2.imshow("img", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("./testJPGImages/test11Result.jpg", image)

from NMS import nms

if __name__ == "__main__":
    ### Test part
    testPeopleJpg = "./testJPGImages/test11.jpg"
    modelWeightPath = "./trainCheckPoint/45696Times.pth"
    boundingBoxesNum = 3
    labelsNum = 1
    S = 8
    imageSizeTest = 1024
    confidenceTh = 0.2

    ### Model
    backbone = ResNet()
    backboneLastChannels = backbone.last_channel
    yoloModel = YoloDetection(backBoneOutChannels=backboneLastChannels, backbone=backbone,
                              BoundingBoxes=boundingBoxesNum, num_classes=labelsNum, SGrid=S, imageSize=imageSizeTest)
    yoloModel.load_state_dict(torch.load(modelWeightPath))
    ### testModel
    yoloModel = yoloModel.eval()
    yoloTrans = YoloTrans(BoundingBoxes=boundingBoxesNum, num_classes=labelsNum, SGrid=S, imageSize=imageSizeTest, device="cpu")
    transforms = tv.transforms.Compose(
         [  tv.transforms.Resize([imageSizeTest, imageSizeTest]),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    pilImg = Image.open(testPeopleJpg).convert("RGB")
    imgTensor = transforms(pilImg).unsqueeze(dim=0)
    # confidence [N, S, S, B],
    # boxes [N, S, S, B * 4], (offsetX, offsetY, offsetW, offsetH)
    # condClasses [N, S, S, NUM_CLASSES]
    preConfidence, preBoxes, preCondClasses = yoloModel(imgTensor)
    ### threshold box select [N, S, S, B]
    confidenceMask = (preConfidence >= confidenceTh).float()
    ### reshape the prediction boxes [N, S, S, B, 4]
    reBoxes = torch.reshape(preBoxes, [-1, S, S, boundingBoxesNum, 4])
    reConfidenceMask = confidenceMask.unsqueeze(dim=-1)
    maskedBoxes = reBoxes * reConfidenceMask
    confidentBoxes = torch.reshape(maskedBoxes, [-1, S , S , boundingBoxesNum * 4])
    ### oriBoxes [N, S * S * Boxes, 4]
    leftTopFormatBoxes = yoloTrans.TransformPredictedBoxes(confidentBoxes)
    boxesMask = torch.reshape(confidenceMask, [-1, S * S * boundingBoxesNum])
    preConfidenceRe = torch.reshape(preConfidence, [-1, S * S * boundingBoxesNum])
    boxesConfidence = []
    boxesCoordinate = []
    for i in range(S * S * boundingBoxesNum):
        if boxesMask[0,i] != 0:
            boxesConfidence.append(preConfidenceRe[0,i])
            boxesCoordinate.append(leftTopFormatBoxes[0, i, :])
    boxesCoordinate = torch.stack(boxesCoordinate,dim=0).detach().cpu().numpy()
    boxesConfidence = torch.stack(boxesConfidence, dim=0).detach().cpu().numpy()
    ### NMS
    picked_boxes , _ = nms(boxesCoordinate, boxesConfidence, threshold=0.1)
    cv2Image = np.array(tv.transforms.Resize([imageSizeTest, imageSizeTest])(pilImg))
    drawBox(np.array(picked_boxes), cv2Image)












