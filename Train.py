import torchvision as tv
from torch.utils.data import DataLoader
import torch
from WarmSch import GradualWarmupScheduler
import torch.cuda.amp as amp
import torch.optim as optim
from DataSet import AssembleDataSet
from Backbone import ResNet
from YoloV1 import YoloDetection
from YoloV1 import YoloLoss
from collections import OrderedDict

if __name__ == "__main__":
    ### config
    device = "cuda:0"
    boundingBoxesNum = 3
    labelsNum = 1
    SGrid = 8
    imageSize = 1024
    coordLambda = 5
    noObjLambda = 0.008
    LR = 1e-5
    multiplier = 100
    reg_lambda = 1e-5
    ### In current version, the batch size only can be 1 !!!
    batchSize = 1
    warmEpoch = 5
    epoch = 25
    displayTimes = 10
    if_loadPre_TrainWeight = True
    preTrainWeightLoadPath = "resnext101_32x8d-8ba56ff5.pth"
    trainCheckPointSavePath = "./trainCheckPoint/"

    ### Model
    backbone = ResNet()
    backboneLastChannels = backbone.last_channel
    yoloModel = YoloDetection(backBoneOutChannels=backboneLastChannels, backbone=backbone,
                              BoundingBoxes=boundingBoxesNum, num_classes=labelsNum, SGrid=SGrid, imageSize=imageSize).to(device)
    yoloModel = yoloModel.train(True)

    ### Data set
    transforms = tv.transforms.Compose(
        [tv.transforms.ToTensor(),
        tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    dataSet = AssembleDataSet('./AssembleDataSet/',transforms , imageSize=imageSize)
    dataLoader = DataLoader(dataSet,batch_size=batchSize, shuffle=True, num_workers=2, pin_memory=True,)

    ### Optimizer
    optimizer = optim.SGD(yoloModel.parameters(), lr=LR, momentum=0.9, weight_decay=reg_lambda, nesterov=True)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epoch, eta_min=0, last_epoch=-1)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=multiplier, total_epoch=warmEpoch,
                                       after_scheduler=cosine_scheduler)
    scaler = amp.GradScaler()

    ### Loss
    lossCri = YoloLoss(coOrd = coordLambda, noObj= noObjLambda, BoundingBoxes=boundingBoxesNum,
                       num_classes=labelsNum, SGrid=SGrid, imageSize=imageSize, device = device).to(device)
    if if_loadPre_TrainWeight:
        preTrainDic = torch.load(preTrainWeightLoadPath)
        newDic = OrderedDict()
        for key, value in preTrainDic.items():
            if "fc" not in key:
                newDic[key] = value
        yoloModel.backbone.load_state_dict(newDic)

    currentTrainingTimes = 0
    for e in range(1, epoch + 1):
        for times , (NImages, NBoxes, NLabels) in enumerate(dataLoader):
            NImages = NImages.to(device)
            #print(NImages.shape)
            NBoxes = NBoxes.to(device)
            #print(NBoxes.shape)
            NLabels = NLabels.to(device)
            #print(NLabels.shape)
            optimizer.zero_grad()
            with amp.autocast():
                ## confidence [N, S, S, B], boxes [N, S, S, B * 4], condClasses [N, S, S, NUM_CLASSES]
                preConfidence, preBoxes, preCondClasses = yoloModel(NImages)
                ordinateLoss, objLoss, noObjLoss, classesLoss = \
                    lossCri(preConfidence=preConfidence,
                               preBoxes=preBoxes,
                               preCondClasses=preCondClasses,
                               groundTruth=NBoxes,
                               groundLabels=NLabels)
            totalLoss = ordinateLoss + objLoss + noObjLoss + classesLoss
            #print(torch.isnan(totalLoss).tolist())
            if torch.isnan(totalLoss).tolist() is False:
                #print("Update")
                scaler.scale(totalLoss).backward()
                scaler.step(optimizer)
                scaler.update()
            currentTrainingTimes += 1
            if currentTrainingTimes % displayTimes == 0 and torch.isnan(totalLoss).tolist() is False:
                with torch.no_grad():
                    print("######################")
                    print("Epoch : %d , Training time : %d" % (e, currentTrainingTimes))
                    print("Total Loss is %.3f " % (totalLoss.item()))
                    print("Coordinate loss is {}".format(ordinateLoss.item()))
                    print("Object confident loss {}".format(objLoss.item()))
                    print("No object confident loss {}".format(noObjLoss.item()))
                    print("Classes Loss {}".format(classesLoss))
                    print("Learning rate is ", optimizer.state_dict()['param_groups'][0]["lr"])
        torch.save(yoloModel.state_dict(),
                   trainCheckPointSavePath + str(currentTrainingTimes) + "Times.pth")
        scheduler.step()

