
for file in results/*.log; do
  python find_best_WER.py ${file} > results/"$(basename "${file/%.*}")".log2
done
