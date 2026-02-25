# DDPM & DDIM: Lightweight Implementation on MNIST (Digit 5)

This repository provides a streamlined implementation of **DDPM** and **DDIM**, specifically designed for **Google Colab**. 
To observe the model's generative behavior in a controlled environment, the training was intentionally focused on the **"digit 5"** class from the MNIST dataset.

## 🚀 Overview
For a deep dive into the overall algorithms and the specific mathematical formulas implemented in this project, please refer to my detailed post on **Medium**.

[🔗 Read the Full Analysis on Medium](YOUR_MEDIUM_ARTICLE_LINK_HERE)

## 💻 How to Run in Google Colab (The Easiest Way)

You don't need to worry about multiple files. Everything you need to run the experiment—from data loading to training and sampling—is integrated into `main.py`.

### **Option 1: Quick Clone (Recommended)**
Run these two lines in a Colab cell to get all files and start immediately:
```bash
!git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
%cd YOUR_REPO_NAME
!python main.py
