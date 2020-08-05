from abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class Mish(nn.Module):

    def __init__(self):
        super(Mish,self).__init__()

    def forward(self,x):
        return x * torch.tanh(F.softplus(x))

class Conv2dDynamicSamePadding(nn.Module):
    """
    The real keras/tensorflow conv2d with same padding
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, groups=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=bias, groups=groups)
        self.stride = self.conv.stride
        self.kernel_size = self.conv.kernel_size
        self.dilation = self.conv.dilation

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]
        h_step = math.ceil(w / self.stride[1])
        v_step = math.ceil(h / self.stride[0])
        h_cover_len = self.stride[1] * (h_step - 1) + 1 + (self.kernel_size[1] - 1)
        v_cover_len = self.stride[0] * (v_step - 1) + 1 + (self.kernel_size[0] - 1)
        extra_h = h_cover_len - w
        extra_v = v_cover_len - h
        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top
        x = F.pad(x, [left, right, top, bottom])
        x = self.conv(x)
        return x


class SeparableConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels=None, norm=True, activation=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        # Q: whether separate conv
        #  share bias between depthwise_conv and pointwise_conv
        #  or just pointwise_conv apply bias.
        # A: Confirmed, just pointwise_conv applies bias, depthwise_conv has no bias.

        self.depthwise_conv = Conv2dDynamicSamePadding(in_channels, in_channels,
                                                      kernel_size=3, stride=1, groups=in_channels, bias=False)
        self.pointwise_conv = Conv2dDynamicSamePadding(in_channels, out_channels, kernel_size=1, stride=1)

        self.norm = norm
        if self.norm:
            # Warning: pytorch momentum is different from tensorflow's, momentum_pytorch = 1 - momentum_tensorflow
            self.bn = nn.BatchNorm2d(out_channels)

        self.activation = activation
        if self.activation:
            self.relu = Mish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.relu(x)

        return x


class YoloDetection(nn.Module, ABC):

    def __init__(self, backBoneOutChannels: int, backbone: nn.Module, BoundingBoxes: int,
                 num_classes: int, SGrid: int, imageSize : int):
        """
        (S x S): Our system divides the input image into an S × S grid. If the center of an object falls into a grid cell,
        that grid cell is responsible for detecting that object.
        (x,y): The (x, y) coordinates represent the center of the box relative to the bounds of the grid cell.
        (w,h): The width and height are predicted relative to the whole image.
        (Confidence): Finally the confidence prediction represents the IOU between the predicted box and any ground truth box.
        It is zero if there is no object in this grid.
        :param backBoneOutChannels:
        :param backbone: 4 times striding operation.
        :param BoundingBoxes:
        :param num_classes:
        :param SGrid:
        """
        super().__init__()
        assert imageSize // 5 != SGrid, "image size must be 5 times bigger than SGrid."
        self.backbone = backbone
        self.B = BoundingBoxes
        self.S = SGrid
        self.nc = num_classes
        self.conv_2 = nn.Sequential(
            Conv2dDynamicSamePadding(in_channels=backBoneOutChannels,out_channels=2048,kernel_size=3,stride=2),
            nn.BatchNorm2d(2048, eps=1e-3, momentum=1-0.99),
            Mish(),
            SeparableConvBlock(2048, 1024, norm=True, activation=True)
        )
        self.conv_1 = nn.Sequential(
            nn.Conv2d(1024, self.B * 5 + self.nc, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.B * 5 + self.nc, eps=1e-3, momentum=1-0.99),
        )

    def forward(self, x):
        """
        :param x:
        :return: confidence [N, S, S, B], boxes [N, S, S, B * 4], condClasses [N, S, S, NUM_CLASSES]
        """
        imgFeature = self.backbone(x)
        imgFeature = self.conv_1(self.conv_2(imgFeature))
        finalOutput = imgFeature.permute(0,2,3,1)
        ### B boxes and condClass
        boxesAndConfidence, condClasses = torch.split(finalOutput, [self.B * 5, self.nc], dim=-1)
        boxes, confidence = torch.split(boxesAndConfidence, [self.B * 4, self.B], dim=-1)
        return torch.sigmoid(confidence), torch.sigmoid(boxes), torch.sigmoid(condClasses)


def compute_iou(box1, box2):
    """Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
    Return:
      (tensor) iou, sized [N,M].
    """

    lt = torch.max(
        box1[:, :2],  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, :2],  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    rb = torch.min(
        box1[:, 2:],  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, 2:],  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    wh = rb - lt  # [N,M,2]
    wh[wh < 0] = 0  # clip at 0
    inter = wh[:, 0] * wh[:, 1]  # [N,M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]

    iou = inter / (area1 + area2 - inter + 1e-4)
    return iou


class YoloLoss(nn.Module, ABC):

    def __init__(self, coOrd, noObj,BoundingBoxes: int,
                 num_classes: int, SGrid: int, imageSize = 448, device = "cpu"):
        super().__init__()
        self.coOrd = coOrd
        self.noObj = noObj
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

    def TransformGroundTruth(self, gtOrdinate):
        """
        Transform yolo coordinate to left, top, right, bottom format.
        :param gtOrdinate: [N, S, S, 4], (x, y, w, h)
        :return: oriBoxes [N, S * S, 4] (x1, y1, x2, y2)
        """
        with torch.no_grad():
            #print(gtOrdinate.device)
            xy, wh = torch.split(gtOrdinate, [2, 2], dim=-1)
            currentLTP = torch.unsqueeze(self.leftTopPositions, dim=-2).unsqueeze(dim=0)  ## [1, S, S, 1, 2]
            xy = torch.reshape(xy, [-1, self.S, self.S, 1, 2]) * self.oneGridDis
            centerXY = xy + currentLTP
            oriWH = wh * self.imageSize
            centerXY = torch.reshape(centerXY, [-1,self.S * self.S, 2])
            oriWH = torch.reshape(oriWH, [-1,self.S * self.S, 2])
            x = centerXY[:,:,0]
            y = centerXY[:,:,1]
            w = oriWH[:,:,0]
            h = oriWH[:,:,1]
            x1 = torch.clamp(x - w / 2.,min=0, max=self.imageSize - 1)
            y1 = torch.clamp(y - h / 2.,min=0, max=self.imageSize - 1)
            x2 = torch.clamp(x + w / 2.,min=0, max=self.imageSize - 1)
            y2 = torch.clamp(y + h / 2.,min=0, max=self.imageSize - 1)
            return torch.stack([x1, y1, x2, y2],dim=-1) #, centerXY

    def objMaskAndEncoder(self, groundTruth, groundLabels):
        """
        In gtOrdinate, if one cell contains object, this cell will contain (x, y, w, h).
        However, if there is no object in the cell, it only contains zeros.
        :param groundTruth: [N, GTs, 4], (x1, y1, x2, y2)
        :param groundLabels:  [N, GTs]
        :return: objMask : [N, S, S], gtOrdinate [N, S, S, 4]; (x, y, w, h), gtLabels [N, S, S, NUM_CLASSES]
        """
        with torch.no_grad():
            bNum = groundTruth.shape[0]
            gtsNum = groundTruth.shape[1]
            x1, y1, x2, y2 = torch.split(groundTruth, [1, 1, 1, 1], dim=-1)
            ### center
            w = x2 - x1
            h = y2 - y1
            ### [N, GTs, 1]
            x = x1 + w / 2.
            y = y1 + h / 2.
            ### judge center in which cell
            ### [N, GTs ,1]
            cellX = torch.floor(x / self.oneGridDis).long()
            cellY = torch.floor(y / self.oneGridDis).long()
            offsetX = (x - cellX * self.oneGridDis) / self.oneGridDis
            offsetY = (y - cellY * self.oneGridDis) / self.oneGridDis
            offsetW = torch.true_divide(w , self.imageSize)
            offsetH = torch.true_divide(h,  self.imageSize)
            ### cat [N, GTs, 4]
            offset = torch.cat([offsetX, offsetY, offsetW, offsetH], dim=-1).float()
            gtOrdinate = torch.zeros([bNum, self.S, self.S, 4]).float().to(self.device)
            gtLabels = torch.zeros([bNum, self.S, self.S, self.nc]).to(self.device)
            for i in range(bNum):
                for j in range(gtsNum):
                    gtOrdinate[i, cellY[i, j, 0], cellX[i, j, 0], :] = offset[i, j, :]
                    gtLabels[i, cellY[i, j, 0], cellX[i, j, 0], groundLabels[i, j]] = 1
            objMask = (gtLabels.sum(dim=-1, keepdim=False) != 0)
            objMask = objMask.float()
            return objMask, gtOrdinate, gtLabels

    def bestIouFind(self, objMask, gtOrdinate, preBoxes):
        """
        :param objMask: objMask : [N, S, S],
        :param gtOrdinate: gtOrdinate [N, S, S, 4] (x, y, w, h)
        :param preBoxes: [N, S, S, BOXES * 4], (x, y, w, h)
        :return: boxObjMask [N, S, S, BOXES], boxIouMaxValue [N, S, S, BOXES]
        """
        bNum = preBoxes.shape[0]
        with torch.no_grad():
            boxObjMask = torch.zeros([bNum, self.S, self.S, self.B]).float().to(self.device)
            boxIouMaxValue = torch.zeros([bNum, self.S, self.S, self.B]).float().to(self.device)
            ## oriBoxes [N, S * S * Boxes * 4] (x1, y1, x2, y2)
            originalBoxes = self.TransformPredictedBoxes(preBoxes).view([-1, self.S, self.S, self.B, 4])
            ## oriGts [N, S * S, 4] (x1, y1, x2, y2)
            originalGts = self.TransformGroundTruth(gtOrdinate).view([-1, self.S, self.S, 4])
            for b in range(bNum):
                for i in range(self.S):
                    for j in range(self.S):
                        if objMask[b, i, j] != 0:
                            currentBoxes = originalBoxes[b, i, j, : , :]
                            #print("current predict boxes {}".format(currentBoxes))
                            currentGts = originalGts[b, i, j, :].unsqueeze(dim=0)
                            #print("gts {}".format(currentGts))
                            ## [b,4] , [1,4] --> [b,1]
                            iouCom = compute_iou(currentBoxes, currentGts).squeeze()
                            #print(iouCom)
                            iouMaxValue, iouMaxIndex = torch.max(iouCom,dim=0)
                            #print("IOU max value {}".format(iouMaxValue))
                            iouMaxIndex = iouMaxIndex.long()
                            boxObjMask[b, i, j, iouMaxIndex] = 1
                            boxIouMaxValue[b, i, j, iouMaxIndex] = iouMaxValue
            return boxObjMask, boxIouMaxValue


    def forward(self,preConfidence, preBoxes, preCondClasses,
                     groundTruth, groundLabels):
        """
        :param preConfidence: [N, S, S, BOXES]
        :param preBoxes: [N, S, S, BOXES * 4], (x, y, w, h)
        :param preCondClasses: [N, S, S, NUM_CLASSES]
        :param groundTruth: [N, GTs, 4], (x1, y1, x2, y2)
        :param groundLabels: [N, GTs]
        :return LOSS
        """
        with torch.no_grad():
            ### objMask : [N, S, S], gtOrdinate [N, S, S, 4] (x, y, w, h), gtLabels [N, S, S, NUM_CLASSES]
            objectMask, gtOrdinate, gtLabels = self.objMaskAndEncoder(groundTruth, groundLabels)
            #print(objectMask)
            #print(groundLabels.shape)
            ### boxObjMask [N, S, S, BOXES], boxIouMaxValue [N, S, S, BOXES]
            boxObjMask, boxIouMaxValue = self.bestIouFind(objectMask, gtOrdinate, preBoxes)
            #print(boxObjMask)

        #######################
        ### coordinate loss ###
        #######################
        ### [N, S, S, B, 4]
        expandPreBoxes = torch.reshape(preBoxes, [-1, self.S, self.S, self.B, 4])
        ### [N, S, S, B, 1]
        boxObjMaskExpand = torch.unsqueeze(boxObjMask,dim=-1)
        ### [N, S, S, B, 2]
        xy, wh = torch.split(expandPreBoxes, [2,2], dim=-1)
        wh = torch.sqrt(wh)
        ### [N, S, S, 2]  --> [N, S, S, 1, 2]
        xyGt, whGt = torch.split(gtOrdinate, [2,2], dim=-1)
        whGt = torch.sqrt(whGt)
        xyGt = torch.unsqueeze(xyGt, dim=-2)
        whGt = torch.unsqueeze(whGt, dim=-2)
        ### [N, S, S, B, 2]
        # print("predict xy {}".format(xy * boxObjMaskExpand))
        # print("GT xy {}".format(xyGt))
        xyLoss = (torch.square(xy - xyGt) * boxObjMaskExpand).sum()
        whLoss = (torch.square(wh - whGt) * boxObjMaskExpand).sum()
        coordinateLoss = (xyLoss + whLoss) * self.coOrd
        #print("coordinate loss {}".format(coordinateLoss))
        #######################
        ### confidence loss ###
        #######################
        noBoxObjMask = 1. - boxObjMask
        # print("Box obj mask {}".format(boxObjMask))
        # print("Non box obj mask {}".format(noBoxObjMask))
        objLoss = (torch.square(preConfidence - boxObjMask) * boxObjMask).sum()
        noObjLoss = self.noObj * (torch.square(preConfidence - boxObjMask) * noBoxObjMask).sum()
        #print("confidence loss {}".format(confidenceLoss))
        ##############################
        ### condition classes loss ###
        ##############################
        classesLoss = (torch.square(preCondClasses - gtLabels).sum(dim=-1, keepdim=False) * objectMask).sum()
        #print("classes loss {}".format(classesLoss))
        return coordinateLoss , objLoss, noObjLoss , classesLoss


import cv2
# drawBox([[x, y, xmax, ymax]], img)
def drawBox(boxes, image):
    """
    :param boxes: np array, [N,4], (x1, y1, x2, y2)
    :param image: np array
    :return:
    """
    numBox = boxes.shape[0]
    for i in range(0, numBox):
        # changed color and width to make it visible
        cv2.rectangle(image, (boxes[i,0], boxes[i,1]), (boxes[i,2], boxes[i,3]), (255, 0, 0), 1)
    cv2.imshow("img", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    testLoss = YoloLoss(1,1, BoundingBoxes=3, num_classes=1, SGrid= 7, imageSize=448)
    testWhiteImg = np.ones([448,448,3]) * 255
    testGts = torch.from_numpy(np.array([[[0, 0, 50, 122], [50, 122, 100, 278], [100, 278, 200, 445]]]))
    testGtLabels = torch.from_numpy(np.array([[0,0,0]]))
    testObjMask, testGtOri, testGtLabels = testLoss.objMaskAndEncoder(testGts,testGtLabels)
    print(testObjMask.shape)
    print(testObjMask)
    print(testGtOri.shape)
    print(testGtOri)
    print(testGtLabels.shape)
    print(testGtLabels)
    drawBoxes = testLoss.TransformGroundTruth(testGtOri).view([7 * 7, 4]).numpy()
    print(drawBoxes.shape)
    drawBox(drawBoxes,testWhiteImg)
    ### test predict
    testInput = torch.rand([1, 7, 7, 3 * 4])
    ### test best iou function
    testObjBoxesMask, testObjIou = testLoss.bestIouFind(testObjMask, testGtOri, testInput)
    print(testObjBoxesMask)
    print(testObjIou)
    ### test mse loss
    mesTestInput = torch.rand([5,5,1])
    mestTestTarget = torch.rand([5,5,5])
    print(F.mse_loss(mesTestInput, mestTestTarget))
    ########
    testInputImage = torch.rand([1, 3, 14, 14]).float()
    testGts = torch.from_numpy(np.array([[[0, 0, 50, 122], [50, 122, 100, 278], [100, 278, 200, 445]]]))
    testGtLabels = torch.from_numpy(np.array([[0,0,0]]))
    testBackBone = nn.Conv2d(3, 1024, 3, 1, 1)
    testYoloDet = YoloDetection(backbone=testBackBone, backBoneOutChannels=1024, BoundingBoxes=3, num_classes=1, SGrid=7, imageSize=448)
    testYoloLoss = YoloLoss(coOrd=5, noObj=0.2, BoundingBoxes=3, num_classes=1, SGrid=7, imageSize=448)
    testConfi, testBoxes, testCondi = testYoloDet(testInputImage)
    print(testConfi.shape)
    print(testBoxes.shape)
    print(testCondi.shape)
    ## preConfidence, preBoxes, preCondClasses, groundTruth, groundLabels
    loss = testYoloLoss(preConfidence = testConfi, preBoxes = testBoxes, preCondClasses = testCondi,
                        groundTruth = testGts, groundLabels = testGtLabels)
    print(loss)
























