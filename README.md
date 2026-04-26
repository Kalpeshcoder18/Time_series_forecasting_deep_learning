# Time-Series Forecasting Pipeline — Roll 102317087

> Course Lab Assignment | TIET  
> Custom GRU-based time-series forecasting built from scratch using PyTorch

---

##  Project Overview

This project implements a complete time-series forecasting pipeline from scratch.

The goal is to understand:
- how sequence data is handled  
- how models use past information  
- where models fail and why  

Two datasets are used:

-  Electricity Dataset (noisy, irregular)
-  Temperature Dataset (smooth, seasonal)

---

##  Models Implemented

| Model | Type |
|------|------|
| MLP | Baseline (no sequence awareness) |
| Custom GRU | From scratch |
| LSTM | Prebuilt |
| Transformer | Prebuilt |

---

##  Personalized Parameters

Roll Number: 102317087  

Digits: 1,0,2,3,1,7,0,8,7  
Sum = 29  

| Parameter | Formula | Value |
|----------|--------|-------|
| window_size | (sum % 10) + 8 | 17 |
| prediction_horizon | (last 2 digits % 3) + 1 | 1 |
| hidden_size | (first 3 digits % 16) + 8 | 14 |
| Model | Last digit odd | Custom GRU |

---

##  How It Works

### 1. Windowing

Convert time-series into supervised learning:

Example:

[10,20,30] → 40  
[20,30,40] → 50  

---

### 2. Custom GRU

GRU maintains memory using hidden state.

Equations:

z = sigmoid(Wz [h, x])  
r = sigmoid(Wr [h, x])  
h~ = tanh(W [r*h, x])  
h = (1-z)h + z h~  

---

### 3. MLP

- Flattens input  
- No sequence understanding  

---

### 4. Train-Test Split

Chronological split (no shuffle)

---

##  Results

### Electricity Dataset

- Noisy and irregular  
- Transformer performs best for small/medium window  
- MLP performs best for large window  
- GRU struggles due to noise  

---

### Temperature Dataset

- Smooth and seasonal  
- All models perform well  
- GRU improves with larger window  
- Differences are small  

---

##  Ablation Study

Window sizes tested:

[8, 17, 34]

---

### Observations

Electricity:
- Small window → less context  
- Large window → MLP best  
- GRU struggles  

Temperature:
- Smooth data → all models good  
- GRU best for large window  

---

##  Failure Analysis

1. Sudden spikes (electricity)  
2. Noise in data  
3. Long dependencies  

---

##  Model Comparison

| Model | Strength | Weakness |
|------|--------|----------|
| MLP | simple, fast | no sequence |
| GRU | memory | struggles with noise |
| LSTM | long memory | slower |
| Transformer | global attention | needs tuning |

---

##  Final Conclusion

Electricity dataset is noisy → models struggle  
Temperature dataset is smooth → models perform well  

Model performance depends on:
- data type  
- window size  

---

##  How to Run

pip install torch numpy pandas matplotlib scikit-learn  

---

##  Dependencies

torch  
numpy  
pandas  
matplotlib  
scikit-learn  

---
