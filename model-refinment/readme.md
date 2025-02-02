# Refine an LLM

# Pre-Planning

- Choose a small open source LLM as the base
- Choose domain specific data to refine (train) the model with
  - Domain specific just means data on something specific, like a job function, or particular product
- Determine the needed base pytorch mechanics needed
  - What pytorch features/things will we need to use
- Determine the technical requirements
- The first trainings until we get positive results I will perform on my dual 3090 Ti system for sake of iteration/speed
  - Ideally we should be able to easily scale the training up and down for various GPUs

# Requirements

- Small base model <= 3B params
- Dataset should come from official documentation, github etc...
  - We will deep dive pytorch XPU, oneAPI, and other Intel ARC ML documentation
- Refining dataset (our new training data) needs to have domain specific knowledge
  - This knowledge should be prepaired in question and answer form
  - We will need at least 10k Q/A pairs
    - We really need around 2500 unique Q/A pairs and upscale this via varied versions of these questions and answers
    - An adjacent LLM will be utilized to both create the initial and the upscale questions from the base data
- Dataset should result in a CVS
- We will refine the model utilize as limit ram as possible while developing on cuda (simply because my system has my RTX 3090 Ti's installed at the time of this writing)
  - We will limit VRAM usage during training to 8GB
  - We will need to utilize checkpointing to minimize VRAM usage
  - We will need to utilize AMP for BF16
    - I want to train this with BF16 if possible since we can utilize that on most modern GPUs and on modern intel CPUs (which doesnt support FP16)
- Memory handling
  - We may need to:
    - Use Quantization (8bit or 4bit) with QLoRA
    - Use Checkpointing to shave off peak vram spikes during back progation
    - Reduce batch sizes and implement gradient accumulation
    - Offload parts to CPU with BF16 (for my 14900K) or FP16 on Zen4/5
  - Estimated VRAM usage during training
    - FP32 (on cpu): 20-40GB
    - BF16/FP16 with AMP: 12-18GB
    - 8-bit Quant QLoRA: 8-9GB
    - 4-bit Quant QLoRA: 5-6GB
    - These estimates are with a small model
  - ARC B580's can handle
    - INT2, INT4, INT8, FP16, BF16, TF32 on tensors (Xe Matrix Extension Compatible)

# Model

Google's open source Gemma 2: <a href="https://huggingface.co/google/gemma-2-2b-it">google/gemma-2-2b-it</a>

# Refinement

We will leverage all the ARC, PyTorch XPU, Tensforflow XPU, oneAPI etc... Intel specific documentation, slides, instructions we can find. A lot of this is in most of the LLMs but its out of date and in most cases not very usable. We will store all this data in the `raw-materials` folder to prepare for conversion for our Dataset.