#!/usr/bin/env bash

# activate conda environment
source /fsx/software/miniconda3/bin/activate lyl

# set environment vars
export FI_PROVIDER=efa
export FI_OFI_RXR_RX_COPY_UNEXP=1
export FI_OFI_RXR_RX_COPY_OOO=1
export FI_EFA_MR_CACHE_ENABLE=1
export FI_OFI_RXR_INLINE_MR_ENABLE=1
export NCCL_DEBUG=INFO

# run
colossalai run \
  --nproc_per_node 8 \
  --host 192.168.2.71,192.168.2.114 \
  --master_addr 192.168.2.71 \
  auto_parallel_with_gpt.py
