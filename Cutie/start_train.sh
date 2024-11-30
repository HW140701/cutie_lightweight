#! /bin/bash
#注意：必须有&让其后台执行，否则没有pid生成   jar包路径为绝对路径
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 nohup torchrun --master_port 25357 --nproc_per_node=1 ./cutie/train.py exp_id=lightweight model=lightweight data=lightweight > ./train_log.txt 2>&1 &

# 将jar包启动对应的进程pid写入文件中，为停止时提供pid
echo $! > ./train_pid.txt