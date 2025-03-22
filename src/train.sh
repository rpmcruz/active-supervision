UNDERCLASSES="2 5 8"
for UNDERCLASS in $UNDERCLASSES; do
    for REP in `seq 1 5`; do
        echo "============ train1.py underclass=$UNDERCLASS rep=$REP ============"
        python3 train1.py student $UNDERCLASS "student$UNDERCLASS-iter0-rep$REP.h5"
        python3 evaluate.py "student$UNDERCLASS-iter0-rep$REP.h5" | tee -a results-accuracy.csv

        for ITER in `seq 1 4`; do
            echo "============ train2.py underclass=$UNDERCLASS rep=$REP iter=$ITER ============"
            python3 train2.py "models/teacher.h5" "student$UNDERCLASS-iter`expr $ITER - 1`-rep$REP.h5" $UNDERCLASS "student$UNDERCLASS-iter$ITER-rep$REP.h5" | tee -a results-nimages.csv
            python3 evaluate.py "student$UNDERCLASS-iter$ITER-rep$REP.h5" | tee -a results-accuracy.csv
        done
    done
done
