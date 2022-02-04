
for file in results/QuartzNet15x5_hu/*.log; do
  python find_best_WER.py ${file} > results/QuartzNet15x5_hu/"$(basename "${file/%.*}")".log2
done
