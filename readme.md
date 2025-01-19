# Welcome to my XPU Notes

This will serve and my notes for setting up and running XPU related projects within PyTorch, Tensorflow, StableDiffusion etc...

I am currently testing with:

- ASRock Challenger ARC A380 6GB
- Acer Predator BiFrost ARC A770 16GB
- Intel ARC B580 Limited Edition 12GB

I am very impressed so far with the Battlemage mixed precision performance. In my benchmarks I am seeing a single B580's performance around an RTX 2080 Ti or Nvidia L4 in Google Colab training a BERT model.

If you are running an ARC GPU in tandem with your iGPU you may run into issues until you disable the iGPU or happen to be running on a MB without iGPU support.

I am far from an expert or engineer, but I do enjoy learning every day.

## Links to instructions
* <a href="Native-Windows-Intel-ARC-XPU-Torch-With Mixed-Precision.md">Native Windows Intel ARC XPU Torch With Mixed Precision</a>
* <a href="Examples/readme.md">Example code with XPU, Mixed Precision, Etc...</a>