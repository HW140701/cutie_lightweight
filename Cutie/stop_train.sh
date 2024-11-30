#! /bin/bash
PID=$(cat ./train_pid.txt)
kill -9 $PID