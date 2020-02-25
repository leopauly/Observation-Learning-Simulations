#! /bin/bash
filename=reach7dof_train_proposed_thesis_reward_eval.py
echo "-------- -Running the file:---------------"
echo $filename

python $filename 0 3 3
python $filename 3 6 3
python $filename 6 10 3

python $filename 0 3 4
python $filename 3 6 4
python $filename 6 10 4
echo "---------------------------------------------"
