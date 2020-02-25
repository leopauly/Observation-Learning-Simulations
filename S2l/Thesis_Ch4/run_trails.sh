#! /bin/bash
filename=reward_eval_task1_train.py
echo "-------- -Running the file:---------------"
echo $filename
python $filename 0  3
python $filename 3  6
python $filename 6  10
echo "---------------------------------------------"
