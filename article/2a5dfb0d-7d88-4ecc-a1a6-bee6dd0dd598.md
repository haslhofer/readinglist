## PowerInfer: A High-Speed Large Language Model Inference Engine on a Consumer-Grade GPU
Summary: PowerInfer accelerates Large Language Model (LLM) inference on a consumer-grade GPU. Its design leverages the observation that a small subset of neurons, called hot neurons, are consistently activated across inputs, while the majority, cold neurons, vary based on input. Hot neurons are preloaded onto the GPU, while cold neurons are computed on the CPU, reducing GPU memory demands and data transfers. Adaptive predictors and neuron-aware sparse operators further optimize efficiency. PowerInfer achieves an average token generation rate of 13.20 tokens/s on an NVIDIA RTX 4090 GPU, only 18% lower than a top-tier server-grade GPU and significantly outperforming existing solutions.

Link: https://arxiv.org/abs/2312.12456

<img src="/img/2a5dfb0d-7d88-4ecc-a1a6-bee6dd0dd598.png" width="400" />
<br/><br/>
