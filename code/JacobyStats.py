import sys

from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter

import SimpleITK as sitk
import atlas_utils
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
        parser.add_argument(
            "-t",
            "--rigidToOrganMask",
            dest="rigidToOrganMask",
            action="store_true",
            help="apply rigid registration to organ mask",
        )
        parser.add_argument(
            "-c",
            "--distanceToConsider",
            dest="considerDistance",
            help="define region around mask to consider",
        )

        args = parser.parse_args()
        if args.mask and args.output:
            sitkMask = sitk.ReadImage(args.mask)
            sitkOrganMask = None
            considerDistance = 0.0
            if args.considerDistance:
                considerDistance = float(args.considerDistance)
            if args.organMask:
                sitkOrganMask = sitk.ReadImage(args.organMask)
            if args.rigid:
                transform = sitk.ReadTransform(args.rigid)
                atlas_utils.applyRigidRegistrationToImgHeader(sitkMask, transform)
                if args.rigidToOrganMask:
                    if sitkOrganMask is not None:
                        atlas_utils.applyRigidRegistrationToImgHeader(sitkOrganMask, transform)
            if args.deformable:
                sitk_displacement_field = sitk.ReadImage(args.deformable)
                jacobi = sitk.DisplacementFieldJacobianDeterminant(sitk_displacement_field)
                jacobiA = sitk.GetArrayFromImage(jacobi)

                resampler = sitk.ResampleImageFilter()
                resampler.SetReferenceImage(sitk_displacement_field)
                resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                resampler.SetDefaultPixelValue(0)
                resampledMask = resampler.Execute(sitkMask)

                sampledMaskA = sitk.GetArrayFromImage(resampledMask)

                file_exists = exists(args.output)
                with open(args.output, "a") as csvFile:
                    w = csv.writer(csvFile)
                    if not file_exists:
                        w.writerow(("FileName", "Structure", "Mean", "Std", "Min", "Max"))

                    jacobyValsInMask = jacobiA[sampledMaskA > 0.0]
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
                        resampledOrganMask = resampler.Execute(sitkOrganMask)
                        sampledOrganMaskA = sitk.GetArrayFromImage(resampledOrganMask)

                        organVals = np.unique(sampledOrganMaskA[sampledMaskA > 0.0])
                        sampledMaskADim = sampledOrganMaskA.ndim
                        mask = np.any(
                            np.expand_dims(sampledOrganMaskA, axis=sampledMaskADim) == organVals, axis=sampledMaskADim
                        )
                        mask[sampledMaskA > 0.0] = False

                        distanceMapTensor = np.ones_like(resampledMask) * -1.0
                        if considerDistance > 0.0:
                            distanceMapSitk = sitk.SignedDanielssonDistanceMap(resampledMask, useImageSpacing=True)
                            distanceMapTensor = sitk.GetArrayFromImage(distanceMapSitk)

                        for organVal in organVals:
                            jacobyValsForOrganOutOfGtv = jacobiA[
                                (sampledOrganMaskA == organVal)
                                & (sampledMaskA == 0.0)
                                & (distanceMapTensor < considerDistance)
                            ]
                            w.writerow(
                                [
                                    args.mask,
                                    f"Organ{organVal.item()} without Mask",
                                    jacobyValsForOrganOutOfGtv.mean().item(),
                                    jacobyValsForOrganOutOfGtv.std().item(),
                                    jacobyValsForOrganOutOfGtv.min().item(),
                                    jacobyValsForOrganOutOfGtv.max().item(),
                                ]
                            )
                            jacobyValsForOrganInGtv = jacobiA[(sampledOrganMaskA == organVal) & (sampledMaskA > 0.0)]
                            w.writerow(
                                [
                                    args.mask,
                                    f"Organ{organVal.item()} AND Mask",
                                    jacobyValsForOrganInGtv.mean().item(),
                                    jacobyValsForOrganInGtv.std().item(),
                                    jacobyValsForOrganInGtv.min().item(),
                                    jacobyValsForOrganInGtv.max().item(),
                                ]
                            )

                        jacobyValsSurroundingOrgan = jacobiA[mask]
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
