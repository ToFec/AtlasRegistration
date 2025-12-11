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

# gtvs=$(find $BASEDIR -maxdepth 3 -type f -name 'GTV*.nrrd') # Frbg
gtvs=$(find $BASEDIR -type f -wholename '*NRRD/HM*res.nrrd') # TUM
#gtvs=$(find $BASEDIR -type d -name 'NRRD' -exec find {} -maxdepth 1 -type f -name 'HM*res.nrrd' \;)
for gtv in $gtvs; do
  imgDir=$(dirname $gtv)
  filename=$(basename $gtv)
  echo $gtv

  # affineRegName="BHFI_vtkMRMLLinearTransformNodeH12Dof.h5" # Frbg
  affineRegName="BHFI_vtkMRMLLinearTransformNodeH12DofContourBased.txt" # TUM

  if [ ! -f "${imgDir}/${affineRegName}" ]; then
    echo "rigid registration file does not exist"
    continue
  fi

  if [ ! -f "${imgDir}/${folderForDeffields}/DefField.mha" ]; then
    echo "deformable registration file does not exist"
    continue
  fi
  python code/ApplyTransformationsToFiles.py -i "${gtv}" -o "${imgDir}/${folderForDeffields}/${filename}" -r "${imgDir}/${affineRegName}" -d "${imgDir}/${folderForDeffields}/DefField.mha" -b

  name="${filename%.*}"
  extension="${filename##*.}"

  python code/ApplyTransformationsToFiles.py -i "${gtv}" -o "${imgDir}/${folderForDeffields}/${name}Rigid.${extension}" -r "${imgDir}/${affineRegName}" -b

  python gtvStats.py -f "${imgDir}/${folderForDeffields}/${name}Rigid.${extension}" -s "${imgDir}/${folderForDeffields}/${filename}" -o "gtvStats${folderForDeffields}TUM.csv"

  echo "done"
done
