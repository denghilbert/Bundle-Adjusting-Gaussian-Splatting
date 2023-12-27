for i in $(seq 2 10)
do
   python train.py -m "output/gates_$i" -s dataset/gates_hall_1x -r $i
   python render.py -m "output/gates_$i" -s dataset/gates_hall_1x
   python metrics.py -m "output/gates_$i" --eval_set "train"
done

