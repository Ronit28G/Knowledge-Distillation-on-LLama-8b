# Optimizing Large Language Models: Profiling and Benchmarking with Fine-Tuning and Knowledge Distillation

## Project Description
This project explores the use of **Knowledge Distillation (KD)** and **Fine-Tuning (FT)** to compress large language models (LLMs) for efficient deployment. We distill the LLaMA 3.1 8B model into a smaller LLaMA 3.2 3B model and fine-tune it on a sentiment classification task. Then, we benchmark the original and compressed models using NVIDIA Nsight tools to analyze their performance and deployment readiness.

## Goal
To evaluate whether multi-stage compression (KD + FT) can enable a 3B model to approach the performance of an 8B model while drastically reducing memory and inference cost.

## Project Milestones 
| Milestone                                         | Status |
|----------------------------------------------------|--------|
| Set up teacher (LLaMA 3.1 8B)                      | ✅     |
| Perform knowledge distillation to LLaMA 3.2 3B     | ✅     |
| Fine-tune the 3B distilled model on sentiment data | ✅     |
| Add 4-bit quantization                             | ✅     |
| Benchmark using Nsight Systems                     | ✅     |   

## Repository Structure (In NYU HPC)
```
.
├── llama-3.1-8B-Instruct/       # Scripts and configs for LLaMA 8B
├── llama-3.2-3B-Instruct/       # Scripts and models for distilled 3B version
├── kd_final_merged/             # Scripts and models for distilled 3B version 
├── profiling/                   # Nsight logs and analysis screenshots
├── hpml_training.ipynb          # Notebook for training and KD
├── evaluation.ipynb             # Notebook for evaluation
```

## Run Instructions

### Switch to an Image containing with correct Nsights library installed

```bash
 /share/apps/images//run-nsight-compute-2023.2.bash
```


### Run Distilled Student Inference
```bash


  python -c /share/apps/pyenv/py3.9/bin/python inference_kd.py
```

### Example Command for Profiling with Nsight
```bash
/ext3/nsight-systems/2023.2.1/bin/nsys profile \
  --trace=cuda,nvtx,cudnn \
  --output=timed_infer_report \
  --force-overwrite true \
  /share/apps/pyenv/py3.9/bin/python inference_kd.py

```


## Sample Demo outputs
<img width="1465" alt="Image" src="https://github.com/user-attachments/assets/3f51007e-14f9-43ee-b192-a65603041fb3" />

## Nsight Systems UI Overview

![Nsight UI Screenshot](https://github.com/user-attachments/assets/27c57f36-41ee-4d99-bd14-cbee78f9a059)

### CUDA API Activity Summary

In the captured 2.345s window, the CUDA API track (located just below the NVTX track) highlights numerous short-duration CUDA function calls. These include:

- `cudaMemcpyAsync` – for asynchronous memory transfers  
- `cudaLaunchKernel` – for launching GPU kernels  
- Memory allocation routines  

#### Color Indicators:

- 🟨 **Yellow ticks** – `cudaMemcpyAsync` (memory copy operations)  
- 🔴 **Red markers** – Synchronization points or memory bottlenecks  
- 🟩 **Green markers** – Kernel launches  
- 🔵 **Blue markers** – Completion or cleanup events  

Most CUDA activity falls under the **NVT inference region**, indicating effective instrumentation and good timing scope coverage.





## Results & Observations

| Metric                         | LLaMA 8B | LLaMA 3B (KD+FT) |
|-------------------------------|----------|------------------|
| HtoD memcpy time              | 114 sec  | 6.9 sec          |
| Data Transfer Volume          | ~16 GB   | ~2.2 GB          |
| Peak Memory Usage             | ~96 GB   | ~28 GB           |
| Kernel Utilization            | GEMM FP32 | kgemm_4bit_inference |
| Token Generation Latency      | ~800 ms  | ~370 ms          |
| Accuracy (Sentiment Task)     |   58.64%   |    58.56%      |

-  ~94% reduction in HtoD time  
-  4-bit quantization enables better kernel scheduling  
-  KD preserves most accuracy despite 5B parameter reduction  
-  Suitable for edge deployment

## CUDA Execution Breakdown
![Image](https://github.com/user-attachments/assets/776118c1-2b92-4025-bc98-05e00cbbce2d)


## Accuracies
![Image](https://github.com/user-attachments/assets/5b68bf6a-4788-46e6-be81-abb9876fc72e)

##  Profiling Tools and Frameworks

- **NVIDIA Nsight Systems** for CUDA profiling

##  Challenges Faced
- Hardware constraints on local GPUs (24GB VRAM)
- Alignment between teacher/student architectures (LLaMA 3.1 → 3.2)
- Profiling overhead and NVTX trace limitations
- Quantization-induced accuracy trade-offs

##  Conclusion
Knowledge Distillation, combined with quantization and fine-tuning, enables substantial compression of large language models while preserving much of their original performance. In our experiments, the LLaMA 3.2 3B student model achieved accuracy comparable to the 8B teacher model, with significantly reduced memory usage and latency. These results highlight the effectiveness of multi-stage model compression techniques for building lightweight, resource-efficient models suitable for environments with limited hardware capacity.

##  This is our final project for ECE-GY 9143 HPML at NYU. 
