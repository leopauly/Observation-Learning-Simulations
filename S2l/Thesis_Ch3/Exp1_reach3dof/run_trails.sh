#! /bin/bash
filename=reach7dof_train_proposed_thesis_reward_eval.py
echo "-------- -Running the file:---------------"
echo $filename


python $filename 0 3 -6
python $filename 3 6 -6
python $filename 6 10 -6



echo "---------------------------------------------"
