#!/bin/bash
# Wait for jlh34 C to finish, then run the remaining probes one after another.
while ! grep -qE '^=== \[[^]]+\] DONE|^STOPPED' results/jlh34_probeC/pipeline.log 2>/dev/null; do sleep 60; done
for rp in "jlh34 A" "jlh32 B" "jlh32 C"; do
  set -- $rp; echo "=== queue: $1 $2 start $(date '+%H:%M')" >> results/queue.log
  REC=$1 PROBE=$2 bash scripts/run_probe.sh --from prep; echo "=== queue: $1 $2 rc=$? $(date '+%H:%M')" >> results/queue.log
done
echo "=== queue: ALL DONE $(date '+%H:%M')" >> results/queue.log
