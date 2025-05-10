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

## Repository Structure
```
.
├── llama_teacher/         # Scripts and configs for LLaMA 8B
├── llama_student/         # Scripts and models for distilled 3B version
├── notebooks/             # Jupyter notebooks for FT and eval
├── profiling/             # Nsight logs and analysis screenshots
├── results/               # Tables and figures from benchmark
├── run.sh                 # Shell script to launch inference
└── requirements.txt       # Python dependencies
```

## Run Instructions

### Setup Environment
```bash
pip install -r requirements.txt
```

### Run Distilled Student Inference
```bash
bash run.sh --model llama_kd --quantized
```

### Example Command for Profiling with Nsight
```bash
nsys profile -o llama_profile ./run.sh
```

## Results & Observations

| Metric                         | LLaMA 8B | LLaMA 3B (KD+FT) |
|-------------------------------|----------|------------------|
| HtoD memcpy time              | 114 sec  | 6.9 sec          |
| Data Transfer Volume          | ~16 GB   | ~2.2 GB          |
| Peak Memory Usage             | ~96 GB   | ~28 GB           |
| Kernel Utilization            | GEMM FP32 | kgemm_4bit_inference |
| Token Generation Latency      | ~800 ms  | ~370 ms          |
| Accuracy (Sentiment Task)     | High     | ~1% drop         |

-  ~94% reduction in HtoD time  
-  4-bit quantization enables better kernel scheduling  
-  KD preserves most accuracy despite 5B parameter reduction  
-  Suitable for edge deployment

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
