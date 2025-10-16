import sys

from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter

import SimpleITK as sitk
import atlas_utils


def main(argv=None):
    if argv is None:
        argv = sys.argv
    else:
        sys.argv.extend(argv)

    try:
        # Setup argument parser
        parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter)
        parser.add_argument(
            "-b",
            "--binary",
            dest="binary",
            action="store_true",
            help="is input a labelmap",
        )
        parser.add_argument(
            "-i",
            "--input",
            dest="input",
            help="inputImage",
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
            "-a",
            "--atlas",
            dest="atlas",
            help="atlas image, serves as a reference",
        )

        args = parser.parse_args()
        if args.input and args.output:
            sitkImage = sitk.ReadImage(args.input)
            if args.rigid:
                transform = sitk.ReadTransform(args.rigid)
                atlas_utils.applyRigidRegistrationToImgHeader(sitkImage, transform)
            if args.deformable:
                if args.atlas:
                    referenceImg = sitk.ReadImage(args.atlas)
                else:
                    referenceImg = sitkImage

                deformation_field = sitk.ReadImage(args.deformable)
                resampler = sitk.ResampleImageFilter()
                resampler.SetReferenceImage(referenceImg)
                if args.binary:
                    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                else:
                    resampler.SetInterpolator(sitk.sitkLinear)
                dis_tx = sitk.DisplacementFieldTransform(
                    sitk.Cast(deformation_field, sitk.sitkVectorFloat64)
                )
                resampler.SetTransform(dis_tx)
                sitkImage = resampler.Execute(sitkImage)
            elif args.atlas:
                resampler = sitk.ResampleImageFilter()
                referenceImg = sitk.ReadImage(args.atlas)
                resampler.SetReferenceImage(referenceImg)
                if args.binary:
                    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                else:
                    resampler.SetInterpolator(sitk.sitkLinear)
                sitkImage = resampler.Execute(sitkImage)

            sitk.WriteImage(sitkImage, args.output, useCompression=True)
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as e:
        raise (e)


if __name__ == "__main__":
    sys.exit(main())
