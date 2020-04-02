#! /bin/bash
filename=reach7dof_train_proposed_thesis_reward_eval.py
echo "-------- -Running the file:---------------"
echo $filename

python $filename 1 3 3 
python $filename 3 6 3
python $filename 6 10 3



echo "---------------------------------------------"
