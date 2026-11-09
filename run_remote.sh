#!/bin/bash
# run_remote.sh  —  exact (bilby + nessai) posteriors of GOLD events on a second machine (see CLAUDE.md)
# --------------------------------------------------------------------------------------------------------
# Events are named by their SNR rank (rank 1 = largest network SNR of the GOLD O5 catalog; inputs/ranking.csv).
# RANKS: e.g. 5,14,26 or 101-150 (default: the ranks of inputs/remote_events.csv).
#
#   bash run_remote.sh check            1. software + likelihood check against Morricone (minutes)  -> must PASS
#   bash run_remote.sh validate         2. sample rank 1 here, compare with Morricone (~0.5-1 h)    -> must PASS
#   bash run_remote.sh run [RANKS]      3. sample the ranks, one at a time, cheapest first, detached (days)
#   bash run_remote.sh status           progress of the running queue
#   bash run_remote.sh export [RANKS]   4. IARA files, corner plots, index, collection: the same code as Morricone
#   bash run_remote.sh pack             5. results_<host>_<date>.tar.gz (+ .sha256) to copy back to Morricone
#
# CPU budget (Hugo, 2026-11-09): HALF of the machine = 10 physical cores. Every sampling run is pinned with taskset
# to 10 distinct physical cores of socket 0 and uses 9 likelihood workers + the nessai main process.
set -e
R=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-$(conda info --base)/envs/gold_fast/bin/python}
export GOLD_CATALOG=$R/inputs/gold_events_slim.hdf5
export GOLD_ASD_DIR=$R/inputs/sensitivity_curves
export GOLD_EXACT_OUT=$R/outputs
export GOLD_REFERENCE_DIR=$R/reference
export GOLD_REPO=$R/code
export GOLD_MPLSTYLE=$R/code/paper.mplstyle
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
NPOOL=9
DEFAULT_RANKS=$(tail -n +2 $R/inputs/remote_events.csv | cut -d, -f1 | paste -sd,)
RANKS=${2:-$DEFAULT_RANKS}
# 10 logical CPUs on 10 different physical cores of socket 0 (no two hyperthreads of the same core)
CPUS=$(lscpu -p=CPU,CORE,SOCKET | grep -v '^#' | awk -F, '$3==0 && !seen[$2]++ {print $1}' | head -10 | paste -sd,)
mkdir -p $R/outputs
# the ranking (all 1023 events) is the same file as Morricone's; the queue rewrites it identically
[ -f $R/outputs/ranking.csv ] || cp $R/inputs/ranking.csv $R/outputs/ranking.csv

case "$1" in
  check)
    for t in latex dvipng kpsewhich; do command -v $t > /dev/null || { echo "MISSING: $t (LaTeX is needed for the corner plots)"; exit 1; }; done
    kpsewhich type1cm.sty > /dev/null || { echo "MISSING: type1cm.sty (texlive-latex-extra)"; exit 1; }
    echo "LaTeX: ok"
    $PY $R/code/validate_remote.py --check-likelihood ;;
  validate)
    echo "rank 1 on CPUs $CPUS with $NPOOL workers"
    taskset -c $CPUS $PY $R/code/run_nessai.py --ranks 1 --npool $NPOOL 2>&1 | tee -a $R/outputs/validate_rank1.log
    $PY $R/code/validate_remote.py --compare-posterior $R/outputs/event_015625 | tee $R/outputs/validate_rank1_compare.txt
    $PY $R/code/export_exact.py --ranks 1 > /dev/null 2>&1 \
      && echo "corner: outputs/figures/corner_rank0001.png (compare with reference/corner_rank0001_morricone.png)" ;;
  run)
    grep -q "POSTERIOR CHECK: PASS" $R/outputs/validate_rank1_compare.txt 2>/dev/null \
      || { echo "validation has not passed (outputs/validate_rank1_compare.txt): not starting production"; exit 1; }
    echo "$(date) queue: ranks $RANKS on CPUs $CPUS with $NPOOL workers" | tee -a $R/outputs/run_remote.log
    # the queue, then the export of the same ranks, detached (survives logout)
    setsid nohup bash -c "taskset -c $CPUS $PY $R/code/run_nessai.py --ranks $RANKS --npool $NPOOL; \
        echo \"\$(date) queue ended, exporting\"; $PY $R/code/export_exact.py --ranks $RANKS" \
        >> $R/outputs/run_remote.log 2>&1 < /dev/null &
    echo "launched (PID $!); follow: bash run_remote.sh status" ;;
  status)
    grep -a "rank .* start\|finished in\|FAILED\|skipped\|ranks\|queue\|exported\|RAILS" $R/outputs/run_remote.log | grep -v INFO | tail -10
    grep -a "n eval" $R/outputs/run_remote.log | tail -1 ;;
  export)
    $PY $R/code/export_exact.py --ranks $RANKS ;;
  pack)
    cd $R/outputs
    T=results_$(hostname)_$(date +%Y%m%d_%H%M).tar.gz
    # raw bilby results (result, setup with this machine's info, log, distance table) of every event run here, the
    # wide-window list, and the exported products; the nessai working folders stay here
    tar czf ../$T --ignore-failed-read *.log validate_rank1*.txt wide_mc.csv exact_index.csv exact_posteriors.hdf5 \
        event_*/bilby_*_result.json event_*/setup.json event_*/bilby_*.log event_*/distance_lookup.npz \
        iara figures 2> /dev/null || true
    cd $R && sha256sum $T > $T.sha256 && ls -la $T $T.sha256 ;;
  *)
    sed -n 2,15p "$0" ;;
esac
