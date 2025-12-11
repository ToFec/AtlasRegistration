#!/bin/bash

RootDir=$(dirname "$0" | xargs -i realpath {})
if [ $# -gt 0 ]; then
  BASEDIR=$(realpath "$1")
else
  BASEDIR=$RootDir
fi

if [ $# -gt 1 ]; then
  folderForDeffields=$2
else
  exit 1
fi

#gtvs=$(find $BASEDIR -maxdepth 3 -type f -name 'GTV*.nrrd') # Frbg
#gtvs=$(find $BASEDIR -maxdepth 3 -type f -name 'HM*res.nrrd') # TUM
gtvs=$(find $BASEDIR -type d -name 'NRRD' -exec find {} -maxdepth 1 -type f -name 'HM*res.nrrd' \;)
for gtv in $gtvs; do
  imgDir=$(dirname $gtv)
  filename=$(basename $gtv)
  echo $gtv

  #affineRegName="BHFI_vtkMRMLLinearTransformNodeH12Dof.h5" # Frbg
  affineRegName="BHFI_vtkMRMLLinearTransformNodeH12DofContourBased.txt" # TUM

  if [ ! -f "${imgDir}/${affineRegName}" ]; then
    echo "rigid registration file ${imgDir}/${affineRegName}  does not exist"
    continue
  fi

  if [ ! -f "${imgDir}/${folderForDeffields}/DefField.mha" ]; then
    echo "deformable registration file does not exist"
    continue
  fi

  python code/JacobyStats.py -m "${gtv}" -r "${imgDir}/${affineRegName}" -d "${imgDir}/${folderForDeffields}/DefField.mha" -o "jacobiStats10mm${folderForDeffields}TUM_tmp.csv" -n "/home/fechter/Bilder/Atlas/seg35_short.nrrd" -c "10.0"

  echo "done"
done
