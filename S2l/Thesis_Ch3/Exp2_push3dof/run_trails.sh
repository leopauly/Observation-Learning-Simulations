#! /bin/bash
filename=push3dofreal_train_proposed_thesis.py
echo "-------- -Running the file:---------------"
echo $filename
python $filename 0  3
python $filename 3  6
python $filename 6  10
echo "---------------------------------------------"
