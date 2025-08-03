#!/bin/bash

if [ -d "./belgium-ts" ]; then
    :
else
    mkdir belgium-ts
    cd belgium-ts
    curl -L -o ./belgium-ts.zip https://www.kaggle.com/api/v1/datasets/download/abhi8923shriv/belgium-ts
    unzip belgium-ts.zip -x "NonTS_TestingBG/*" "Seqs_poses_annotations*" "BelgiumTSD_annotations*"   
    rm belgium-ts.zip

fi
