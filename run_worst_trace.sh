BP="$1"
rm -rf OUTDIR/*
./compile cbp -DPREDICTOR="$BP<>"
./cbp ./traces/gmsh-1.30841_0_trace.gz test 1000000 40000000 > OUTDIR/${BP}.out
./predictor_metrics.py OUTDIR | ./vfs.py
awk -F, 'NR==1 {print 1000 * $9 / $2}' OUTDIR/${BP}.out
awk -F, 'NR==1 {print $12}' OUTDIR/${BP}.out