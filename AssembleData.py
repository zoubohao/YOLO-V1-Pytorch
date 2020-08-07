import re
import os
from shutil import copyfile

class INRIAPerson(object):

    def __init__(self, root, outputDir):
        self.images = list(sorted(os.listdir(os.path.join(root,"pos"))))
        self.imagePath = os.path.join(root, "pos")
        self.annotPath = os.path.join(root, "annotations")
        self.outputAnnoPath = os.path.join(outputDir, "Annotations")
        self.outputImagesPath = os.path.join(outputDir, "Images")


    @staticmethod
    def readAndWriteAnnotation(file_path, output_path):
        regex = re.compile("\(\d+, \d+\) - \(\d+, \d+\)")
        numberFind = re.compile("\d+")
        with open(file_path, "r") as rh, open(output_path, "w") as wh:
            for line in rh:
                oneLine = line.strip()
                findResult = regex.findall(oneLine)
                if len(findResult) != 0:
                    for ordinate in findResult:
                        numFind = numberFind.findall(ordinate)
                        for num in numFind:
                            wh.write(num + "\t")
                        wh.write("\n")

    def run(self):
        for fileName in self.images:
            currentName = str(fileName.split(".")[0])
            copyfile(os.path.join(self.imagePath, currentName + ".png"),
                     os.path.join(self.outputImagesPath, currentName  + ".jpg"))
            self.readAndWriteAnnotation(os.path.join(self.annotPath, currentName + ".txt"),
                                        os.path.join(self.outputAnnoPath, currentName + ".txt"))


class PennFudanPed(object):

    def __init__(self, root, outputDir):
        self.images = list(sorted(os.listdir(os.path.join(root,"PNGImages"))))
        self.imagePath = os.path.join(root, "PNGImages")
        self.annotPath = os.path.join(root, "Annotation")
        self.outputAnnoPath = os.path.join(outputDir, "Annotations")
        self.outputImagesPath = os.path.join(outputDir, "Images")



    @staticmethod
    def readAndWriteAnnotation(file_path, output_path):
        regex = re.compile("\(\d+, \d+\) - \(\d+, \d+\)")
        numberFind = re.compile("\d+")
        with open(file_path, "r") as rh, open(output_path, "w") as wh:
            for line in rh:
                oneLine = line.strip()
                findResult = regex.findall(oneLine)
                if len(findResult) != 0:
                    for ordinate in findResult:
                        numFind = numberFind.findall(ordinate)
                        for num in numFind:
                            wh.write(num + "\t")
                        wh.write("\n")

    def run(self):
        for fileName in self.images:
            currentName = str(fileName.split(".")[0])
            copyfile(os.path.join(self.imagePath, currentName + ".png"),
                     os.path.join(self.outputImagesPath, currentName + ".jpg"))
            self.readAndWriteAnnotation(os.path.join(self.annotPath, currentName + ".txt"),
                                        os.path.join(self.outputAnnoPath, currentName + ".txt"))

class ETH(object):

    def __init__(self, root, output_path, ):
        self.root = root
        self.diName = self.root.split("/")[-1]
        self.outputPath = output_path
        self.files = list(sorted(os.listdir(root)))


    def processAnnotation(self, fileName):
        imageNameRe = re.compile("image_\d+?_?\d?\.png")
        boxesRe = re.compile("\(\d+,.*?\d+,.*?\d+,.*?\d+\)")
        numberFind = re.compile("\d+")
        with open(os.path.join(self.root, fileName), mode="r") as rh:
            for oneLine in rh:
                line = oneLine.strip("\n")
                print(line)
                print(imageNameRe.findall(line))
                imageName = imageNameRe.findall(line)[0].split(".")[0]
                boxes = boxesRe.findall(line)
                with open(os.path.join(self.outputPath, "Annotations", self.diName + imageName + ".txt"), mode="w") as wh:
                    for box in boxes:
                        nums = numberFind.findall(box)
                        for num in nums:
                            wh.write(num + "\t")
                        wh.write("\n")

    def run(self):
        for name in self.files:
            if "txt" in name:
                self.processAnnotation(name)
            else:
                currentName = str(name.split(".")[0])
                copyfile(os.path.join(self.root, currentName + ".png"),
                         os.path.join(self.outputPath, "Images" ,self.diName + currentName + ".jpg"))



if __name__ == "__main__":
    # inrTrain = INRIAPerson("INRIAPerson/Train", "AssembleDataSet")
    # inrTrain.run()
    # inrTest = INRIAPerson("INRIAPerson/Test", "AssembleDataSet")
    # inrTest.run()
    # fudan = PennFudanPed("./PennFudanPed", "./AssembleDataSet")
    # fudan.run()
    # ethInsSunny = ETH("./ETH/sunny", "./AssembleDataSet")
    # ethInsSunny.run()
    ethInsSunny = ETH("./ETH/JEL", "./AssembleDataSet")
    ethInsSunny.run()




















