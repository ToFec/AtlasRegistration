import numpy as np
import sys, getopt
from os.path import exists
import csv
import SimpleITK as sitk


def rearangeIdxVector(x, y, z):
    return (int(x), int(y), int(z))


def main(argv):
    try:
        opts, _ = getopt.getopt(argv, "f:s:o:", ["firstFile=", "secondFile=", "outputFile="])
    except getopt.GetoptError as e:
        print(e)
        return

    fileName = None
    outputFileName = "gtvStats.csv"
    secondFileName = None

    for opt, arg in opts:
        if opt in ("--inputFiles", "-f"):
            fileName = str(arg)
        elif opt in ("--outputFile", "-o"):
            outputFileName = str(arg)
        elif opt in ("--secondFile", "-s"):
            secondFileName = str(arg)

    if fileName is not None and secondFileName is not None:
        itkImg0 = sitk.ReadImage(fileName)
        imgArray0 = sitk.GetArrayFromImage(itkImg0)

        itkImg1 = sitk.ReadImage(secondFileName)
        imgArray1 = sitk.GetArrayFromImage(itkImg1)        

        imgSpacing0 = itkImg0.GetSpacing()
        volume0 = len(imgArray0[imgArray0 > 0]) * imgSpacing0[0] * imgSpacing0[1] * imgSpacing0[2]
        
        imgSpacing1 = itkImg1.GetSpacing()
        volume1 = len(imgArray1[imgArray1 > 0]) * imgSpacing1[0] * imgSpacing1[1] * imgSpacing1[2]        


        file_exists = exists(outputFileName)
        f = open(outputFileName, "a")
        w = csv.writer(f)
        if not file_exists:
            w.writerow(("FileName0", "FileName1", "Volume0", "Volume1", "VolumeRatio"))

        w.writerow([fileName, secondFileName, str(volume0), str(volume1), str(volume0 / volume1)])

    f.close()


if __name__ == "__main__":
    main(sys.argv[1:])

