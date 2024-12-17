import sys

from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter

import SimpleITK as sitk
import atlas_utils
from imageTransformation import Transformation
import torch
import numpy as np

import csv
from os.path import exists


def main(argv=None):
    if argv is None:
        argv = sys.argv
    else:
        sys.argv.extend(argv)

    try:
        # Setup argument parser
        parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter)
        parser.add_argument(
            "-m",
            "--mask",
            dest="mask",
            help="inputMask",
        )
        parser.add_argument(
            "-r",
            "--rigid",
            dest="rigid",
            help="rigid Deformation",
        )
        parser.add_argument(
            "-d",
            "--def",
            dest="deformable",
            help="deformable Deformation",
        )
        parser.add_argument(
            "-o",
            "--output",
            dest="output",
            help="output file",
        )
        parser.add_argument(
            "-n",
            "--OrganMask",
            dest="organMask",
            help="second Mask",
        )

        args = parser.parse_args()
        if args.mask and args.output:
            sitkMask = sitk.ReadImage(args.mask)
            sitkOrganMask = None
            if args.organMask:
                sitkOrganMask = sitk.ReadImage(args.organMask)
            if args.rigid:
                transform = sitk.ReadTransform(args.rigid)
                atlas_utils.applyRigidRegistrationToImgHeader(sitkMask, transform)
                if sitkOrganMask is not None:
                    atlas_utils.applyRigidRegistrationToImgHeader(sitkOrganMask, transform)
            if args.deformable:
                defField = atlas_utils.loadDefField(args.deformable)

                defFieldITK = sitk.ReadImage(str(args.deformable))
                defFieldSpacing = defFieldITK.GetSpacing()

                jacobi = atlas_utils.jacobianDeterminant(defField, defFieldSpacing)

                imgMaskA = sitk.GetArrayFromImage(sitkMask)
                imgMaskA = imgMaskA.transpose()
                imgMaskA = torch.from_numpy(imgMaskA.astype(np.float32))[None, None, ...]

                transformer = Transformation()
                transformer.setIdentityTransform(defField.shape)
                idTransform = transformer.identityTransform
                sampledMaskA = transformer.sampleImage(imgMaskA, idTransform[None, ...], interpolationType="nearest")

                file_exists = exists(args.output)
                with open(args.output, "a") as csvFile:
                    w = csv.writer(csvFile)
                    if not file_exists:
                        w.writerow(("FileName", "Structure", "Mean", "Std", "Min", "Max"))

                    jacobyValsInMask = jacobi[sampledMaskA > 0.0]
                    w.writerow(
                        [
                            args.mask,
                            "mask",
                            jacobyValsInMask.mean().item(),
                            jacobyValsInMask.std().item(),
                            jacobyValsInMask.min().item(),
                            jacobyValsInMask.max().item(),
                        ]
                    )

                    if sitkOrganMask is not None:
                        imgOrganMaskA = sitk.GetArrayFromImage(sitkOrganMask)
                        imgOrganMaskA = imgOrganMaskA.transpose()
                        imgOrganMaskA = torch.from_numpy(imgOrganMaskA.astype(np.float32))[None, None, ...]
                        sampledOrganMaskA = transformer.sampleImage(
                            imgOrganMaskA, idTransform[None, ...], interpolationType="nearest"
                        )
                        organVals = torch.unique(sampledOrganMaskA[sampledMaskA > 0.0])
                        sampledMaskADim = sampledOrganMaskA.dim()
                        mask = (sampledOrganMaskA.unsqueeze(sampledMaskADim) == organVals).any(dim=sampledMaskADim)

                        for organVal in organVals:
                            jacobyValsForOrgan = jacobi[sampledOrganMaskA == organVal]
                            w.writerow(
                                [
                                    args.mask,
                                    organVal,
                                    jacobyValsForOrgan.mean().item(),
                                    jacobyValsForOrgan.std().item(),
                                    jacobyValsForOrgan.min().item(),
                                    jacobyValsForOrgan.max().item(),
                                ]
                            )

                        jacobyValsSurroundingOrgan = jacobi[mask]
                        w.writerow(
                            [
                                args.mask,
                                "allOrgans",
                                jacobyValsSurroundingOrgan.mean().item(),
                                jacobyValsSurroundingOrgan.std().item(),
                                jacobyValsSurroundingOrgan.min().item(),
                                jacobyValsSurroundingOrgan.max().item(),
                            ]
                        )

        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as e:
        raise (e)


if __name__ == "__main__":
    sys.exit(main())
