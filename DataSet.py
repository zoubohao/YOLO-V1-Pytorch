import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import dataset
import torchvision as tv


class AssembleDataSet(dataset.Dataset):
    def __init__(self, root, transforms, imageSize = 448):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "Images"))))
        print("There are {} images in this data set.".format(len(self.imgs)))
        self.img2Boxes = list()
        for imgName in self.imgs:
            currentName = str(imgName.split(".")[0])
            currentImageBoxes = list()
            with open(os.path.join(root, "Annotations", currentName + ".txt"), "r") as rh:
                for line in rh:
                    oneLine = line.strip("\n").split("\t")
                    currentImageBoxes.append([float(oneLine[0]), float(oneLine[1]), float(oneLine[2]), float(oneLine[3])])
            self.img2Boxes.append(currentImageBoxes)

        self.imageSize = imageSize
        self.resize = tv.transforms.Resize([imageSize,imageSize])

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "Images", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        imgNp = np.array(img)
        # get bounding box coordinates for each mask
        num_objs = len(self.img2Boxes[idx])
        boxes = self.encoder(imgNp, self.img2Boxes[idx])
        # convert everything into a torch.Tensor
        # boxes (FloatTensor[Gts, 4])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        # labels (Int64Tensor[Gts]): the label for each bounding box
        labels = torch.zeros((num_objs,), dtype=torch.int64)

        resizedImg = self.resize(img)

        ### if transforms
        if self.transforms is not None:
            resizedImg = self.transforms(resizedImg)

        ### tensor, boxes: [Gts, 4], labels: [Gts]
        return resizedImg, boxes, labels

    def __len__(self):
        return len(self.imgs)

    def encoder(self, imageNp, boxes):
        y_ = imageNp.shape[0]
        x_ = imageNp.shape[1]
        x_scale = self.imageSize / x_
        y_scale = self.imageSize / y_
        encodeBoxes = []
        for i in range(len(boxes)):
            origLeft, origTop, origRight, origBottom = boxes[i]
            x1 = int(np.round(origLeft * x_scale))
            y1 = int(np.round(origTop * y_scale))
            x2 = int(np.round(origRight * x_scale))
            y2 = int(np.round(origBottom * y_scale))
            if abs(x2 - x1) == 0 :
                x2 = x1 + 10
            if abs(y2 - y1) == 0:
                y2 = y1 + 10
            encodeBoxes.append([x1, y1, x2, y2])
        return encodeBoxes

import cv2
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
    import torchvision
    dataSet = AssembleDataSet('./AssembleDataSet/',torchvision.transforms.ToTensor())
    resizeImg , reBoxes, reLabels = dataSet.__getitem__(1000)
    resizeImg = np.array(tv.transforms.ToPILImage()(resizeImg))
    print(resizeImg.shape)
    print(reBoxes.shape)
    print(reLabels.shape)
    drawBox(np.array(reBoxes), resizeImg)