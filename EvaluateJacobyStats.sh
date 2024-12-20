#!/bin/bash

RootDir=$(dirname "$0" | xargs -i realpath {})
if [ $# -gt 0 ]
 then
 BASEDIR=$(realpath "$1")
 else
 BASEDIR=$RootDir
fi

if [ $# -gt 1 ]
 then
 folderForDeffields=$2
 else
 exit 1
fi


gtvs=$(find $BASEDIR -maxdepth 3 -type f -name 'GTV*.nrrd')
for gtv in $gtvs
do
    imgDir=$(dirname $gtv)
    filename=$(basename $gtv)
    echo $gtv
    
    if [ ! -f "${imgDir}/BHFI_vtkMRMLLinearTransformNodeH12Dof.h5" ]; then
    	#echo "rigid registration file does not exist"
    	continue
    fi
    
    if [ ! -f "${imgDir}/${folderForDeffields}/DefField.mha" ]; then
   	 #echo "deformable registration file does not exist"
   	 continue 	
    fi
    
    python code/JacobyStats.py -m "${gtv}" -r "${imgDir}/BHFI_vtkMRMLLinearTransformNodeH12Dof.h5" -d "${imgDir}/${folderForDeffields}/DefField.mha" -o "jacobiStats${folderForDeffields}.csv" -n "/home/fechter/Bilder/Atlas/seg4_short.nrrd"
    
    echo "done"
done
