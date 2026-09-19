# Reproducing

Every number reported in this project, and how to regenerate it.

## Environment

All measurements were taken on this configuration. GPU model matters: it is the
same class used in Im2win's own published evaluation, which is what makes the
reproduction directly comparable.

| Component | Version |
|---|---|
| GPU | NVIDIA RTX 3090, 24GB |
| Driver | 595.71.05 |
| CUDA toolkit | 12.8 |
| PyTorch | 2.11.0+cu128 |
| Python | 3.12 |
| Architecture target | sm_86 |

cuDNN shows genuine run to run variance in algorithm selection on this
hardware, between 17.6 and 23.8 TFLOPS on the same shape across fresh
processes. Every comparison here reports the median of repeated trials, and
ratios are taken against the cuDNN figure most favorable to cuDNN, which is
the conservative choice.

## Dependencies

```bash
pip install torch torchvision torchaudio   # match the CUDA version above
pip install monai fvcore nibabel
git clone https://github.com/MIC-DKFZ/MedNeXt.git && cd MedNeXt && pip install -e .
```

Do not install the `nnunetv2` package. MedNeXt's task format, planner, and
trainers are nnU-Net v1, and v2's planner classes do not exist in that codebase.

MedNeXt ships one bug that blocks preprocessing:
`nnunet_mednext/experiment_planning/experiment_planner_baseline_3DUNet.py`
hardcodes `current_module="nnunet.preprocessing"`, a stale reference from
before the fork was renamed. Change it to `nnunet_mednext.preprocessing`.

## Kernel level results, no dataset required

These produce the central findings and need only a GPU.

```bash
cd bench
python verify_correctness.py         # run first, always
python bench_reproduce_im2win.py     # Section 3: 1.56x kernel only, 1.13x honest
python bench_crossover.py            # Section 6.5: the crossover map
```

For the published kernel's own throughput, build their benchmark with the
parameterized harness:

```bash
cd /path/to/im2win
export CUDA_SOURCE_NAME=im2win_async_128x64_4_base_conv9.cu
mkdir build && cd build
cmake ../tensor_core_tflops -G Ninja && ninja
./sweep 256 64 56 56 64 3 1     # B Cin H W Cout K stride
```

The harness reports nonzero output count alongside throughput. This is not
optional. The released kernels return silently zero or partially computed
output at any shape other than their own benchmark configuration, and timing
an empty launch produces throughput figures around 190 times device peak.

## Dataset results

```bash
export nnUNet_raw_data_base=/workspace/nnUNet_raw_data_base
export nnUNet_preprocessed=/workspace/nnUNet_preprocessed
export RESULTS_FOLDER=/workspace/RESULTS_FOLDER
export nnUNet_n_proc_DA=12

# Medical Segmentation Decathlon Liver, 27GB. 131 labeled training volumes,
# 70 unlabeled test volumes, three classes.
curl -L "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task03_Liver.tar" -o Task03_Liver.tar
tar -xf Task03_Liver.tar && rm Task03_Liver.tar

python nnunet_mednext/experiment_planning/nnUNet_convert_decathlon_task.py -i Task03_Liver
mednextv1_plan_and_preprocess -t 3 \
  -pl3d ExperimentPlanner3D_v21_customTargetSpacing_1x1x1 -pl2d None -tf 3
```

`-tf 3` is required. The default of 8 full-resolution workers exhausts memory
on the nine largest volumes in this dataset and deadlocks, producing processes
that stay alive while accumulating no CPU time. Diagnose stalls by comparing
file write timestamps, not by checking whether processes exist.

Training then needs `--use_compressed_data`, because the unpack step converts
npz to npy and requires roughly 85GB for 131 volumes:

```bash
mednextv1_train 3d_fullres nnUNetTrainerV2_150ep 3 0 \
  -p nnUNetPlansv2.1_trgSp_1x1x1 --use_compressed_data
cd bench && python bench_end_to_end.py      # latency, memory, profile
cd ../src && python dice_compare.py          # accuracy on identical weights
```

`nnUNetTrainerV2_150ep` is a trainer subclass setting `max_num_epochs = 150`.
A reduced schedule is appropriate because the kernel comparison runs on
identical weights, so absolute Dice need not match published state of the art.

## Traps

Three cost real time if rediscovered.

**Silent wrongness.** The failure mode of the kernels studied here is returning
zeros without raising. Verify output, never trust timing alone.

**TF32.** Enabled by default on Ampere. Its 10 bit mantissa produces relative
errors near 3e-04 that look like algorithmic bugs. Disable it when verifying
correctness, or precision gets misattributed to correctness.

**Single measurements.** Three separate headline numbers in this project had to
be corrected after repeated trials contradicted a single reading, including one
that would have inverted a conclusion. Always report medians of repeated trials.
