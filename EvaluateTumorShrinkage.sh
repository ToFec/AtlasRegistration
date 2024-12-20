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
    python /home/fechter/workspace/AtlasRegistration/aladdin/code/ApplyTransformationsToFiles.py -i "${gtv}" -o "${imgDir}/${folderForDeffields}/${filename}" -r "${imgDir}/BHFI_vtkMRMLLinearTransformNodeH12Dof.h5" -d "${imgDir}/${folderForDeffields}/DefField.mha" -b
    
    name="${filename%.*}"
    extension="${filename##*.}"
    
    python /home/fechter/workspace/AtlasRegistration/aladdin/code/ApplyTransformationsToFiles.py -i "${gtv}" -o "${imgDir}/${folderForDeffields}/${name}Rigid.${extension}" -r "${imgDir}/BHFI_vtkMRMLLinearTransformNodeH12Dof.h5" -b
    
    python gtvStats.py -f "${imgDir}/${folderForDeffields}/${name}Rigid.${extension}" -s "${imgDir}/${folderForDeffields}/${filename}" -o "gtvStats${folderForDeffields}.csv"
    
    echo "done"
done
