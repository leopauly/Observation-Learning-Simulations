#! /bin/bash
filename=push3dofreal_train_proposed_thesis_reward_eavl.py
echo "-------- -Running the file:---------------"
echo $filenam
python $filename 0  3 0
python $filename 3  6 0
python $filename 6  10 0

echo "---------------------------------------------"
