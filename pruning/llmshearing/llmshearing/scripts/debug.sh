source ../configs/setup.sh
check_sbash debug_%j 16 128 6 1

# export end_time="2025-05-08T10:00:00"
# e.g. 2025-05-07T10:00:00
# python utils/debug.py --end-time ${end_time}
nohup python /scratch/yx3038/Research/pruning/hpc-dark-magic/keep_alive.py \
    --interval 1 --size 512 \
     > keepalive.log 2>&1 &

echo "Sleeping..."
sleep 2d