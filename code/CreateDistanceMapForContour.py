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
            "-i",
            "--input",
            dest="input",
            help="inputImage",
        )
        parser.add_argument(
            "-o",
            "--output",
            dest="output",
            help="output file",
        )
        parser.add_argument("-m", "--maxDist", dest="maxDist", help="maximum distance to consider", default=200.0)

        args = parser.parse_args()
        if args.input and args.output:
            sitkImage = sitk.ReadImage(args.input)
            sigendDistanceMapTensor = atlas_utils.createSignedDistanceMap(sitkImage, args.maxDist)
            sitkOutputImage = sitk.GetImageFromArray(sigendDistanceMapTensor.squeeze())
            sitkOutputImage.SetDirection(sitkImage.GetDirection())
            sitkOutputImage.SetOrigin(sitkImage.GetOrigin())
            sitkOutputImage.SetSpacing(sitkImage.GetSpacing())
            sitk.WriteImage(sitkOutputImage, args.output, True)
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as e:
        raise (e)


if __name__ == "__main__":
    sys.exit(main())
