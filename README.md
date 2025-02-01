# Deep Learning in xxx 4090 Hours  

(this `xxx` will become an actual number when a number of tasks are completed)

## 🎯 Our Goal  

When studying Deep Learning, mastering **implementation skills** and understanding **practical details** are crucial. However, there is a lack of a **comprehensive** codebase that covers fundamental methods in an accessible way. Additionally, many existing implementations focus on large-scale datasets, making them difficult for individuals to follow and experiment with.  

This repository aims to address these challenges by providing a **well-organized**, **beginner-friendly**, and **comprehensive** collection of fundamental Deep Learning implementations. While our code may not achieve best results and primarily runs on toy datasets (e.g., MNIST), we prioritize **clarity and ease of use**. Specifically, we ensure that all implementations can be executed on a **single NVIDIA RTX 4090 GPU** within a reasonable amount of time.  

Without specific mentioning, all of our implementations are provided in **Jupyter notebooks (`.ipynb`)**, with clear explanations and detailed comments to facilitate learning.

## 👫 Join Us!  

We welcome contributions from anyone passionate about helping more Deep Learning enthusiasts get started! Our goal is to utilize the power of the GitHub community and make learning more accessible.  

### Creating a New Task

To contribute your code for a new task, create a new branch named **`yourname.task`** (e.g., `zhh.resnet`) **from branch [dev_start](https://github.com/Hope7Happiness/Deep-Learning/tree/dev_start)**. This branch is intentionally kept simple to avoiding downloading lots of files during cloning.

Your branch can include experimental or development code, but only finalized notebooks will be merged into the `main` branch. If your code requires additional dependencies, please include a `requirements.txt` file.

Please ensure your code is **concise, easy to understand, and runnable on a single 4090 GPU**. You can refer to the example code at [here](TODO). 

After completing your task, submit an **issue** (not pull request, as your branch doesn't exist in the main repo), which include a link to your branch. We will merge it into the same branch name, but in our main repo. After that, we will check your code and merge it into the `main` branch. If you made further modifications, you can then submit a pull request to the corresponding branch in the main repo.

### Examining Existing Tasks

(To be written)

## 🛠️ Installation

The code should be generally runable on **Python 3.10 + Pytorch 2.2**. For specific tasks, you may have to install additional dependencies based on its instructions.

## 📖 Table Of Contents

### Image Classification

[ ] Residual Networks: [paper](...), [code](...), **cost**=1 4090 hour

### Image Generation

#### Traditional Methods

[ ] VAE

[ ] GAN

#### Modern Generative Models

[ ] DDPM

[ ] Flow Matching

### Text Generation

### Representation Learning

[ ] SimCLR

## Notes for Developers

(Only for repository managers)

### Merge a new branch into main repository:

```shell
git checkout -b name.task dev_start
git pull https://URL.git name.task
git push --set-upstream origin name.task
```