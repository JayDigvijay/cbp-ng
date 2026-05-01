BP="$1"
rm -rf OUTDIR/*
./compile cbp -DPREDICTOR="$BP"
./cbp ./traces/505-mcf-1_45923_trace.gz test 1000000 40000000 > OUTDIR/${BP}.out
./predictor_metrics.py OUTDIR | ./vfs.py
awk -F, 'NR==1 {print 1000 * $9 / $2}' OUTDIR/${BP}.out