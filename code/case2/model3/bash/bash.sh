#!/bin/bash

cd /home/wuyan/dygmamba_project/Real/Claude/Other/code/plan1/case3/ananlysis/Bcell_src/bash

jid1=$(sbatch data_process.sh | awk '{print $4}')
echo "CPU job submitted: $jid1"

jid2=$(sbatch --dependency=afterok:$jid1 dyg_bash.sh | awk '{print $4}')
echo "GPU job submitted: $jid2, waiting for $jid1"