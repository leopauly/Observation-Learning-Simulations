#! /bin/bash
filename=reach7dof_train_proposed_thesis_reward_eval_trajectorymaps.py
echo "-------- -Running the file:---------------"
echo $filename


##python $filename 0 3 -4 flatten_1/Reshape:0   #x
#python $filename 3 6 -4 flatten_1/Reshape:0  #
#python $filename 6 10 -4 flatten_1/Reshape:00 #flatten_1/Reshape:0

#python $filename 0 3 -1 dropout_1/cond/Merge:0  #flatten_1/Reshape:0
#python $filename 3 6 -1 dropout_1/cond/Merge:0 #flatten_1/Reshape:0#
#python $filename 6 10 -1 dropout_1/cond/Merge:0 #flatten_1/Reshape:0


#python $filename 0 3 -100 flatten_1/Reshape:0   #flatten_1/Reshape:0
#python $filename 3 6 -100 flatten_1/Reshape:0  #flatten_1/Reshape:0
#python $filename 6 10 -100 flatten_1/Reshape:0  #flatten_1/Reshape:0

#python $filename 0 3 -2 flatten_1/Reshape:0   #flatten_1/Reshape:0
#python $filename 3 6 -2 flatten_1/Reshape:0  #flatten_1/Reshape:0
#python $filename 6 10 -2 flatten_1/Reshape:0  #flatten_1/Reshape:0

python $filename 0 3 0 flatten_1/Reshape:0   #flatten_1/Reshape:0
python $filename 3 6 0 flatten_1/Reshape:0  #flatten_1/Reshape:0
python $filename 6 10 0 flatten_1/Reshape:0  #flatten_1/Reshape:0

echo "---------------------------------------------"
