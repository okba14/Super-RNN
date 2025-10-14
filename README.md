<!-- PROJECT LOGO -->
<p align="center">
  <img src="https://img.shields.io/badge/AI%20SECURITY%20ENGINE-KALI%20LINUX-black?style=for-the-badge&logo=linux&logoColor=white"/>
  <img src="https://img.shields.io/badge/RNN%20From%20Scratch-C%20Language-blue?style=for-the-badge&logo=c&logoColor=white"/>
  <img src="https://img.shields.io/badge/Platform-Linux%20%7C%20Security-red?style=for-the-badge&logo=kalilinux&logoColor=white"/>
</p>

<h1 align="center">⚙️ Super RNN: Advanced AI System for Kali Linux Security Analysis ⚙️</h1>

<p align="center">
  <em>Custom-built Recurrent Neural Network (RNN) in pure C for advanced threat prediction, system learning, and security intelligence.</em>
</p>

<p align="center">
  <a href="https://github.com/okba14"><img src="https://img.shields.io/github/stars/okba14?style=flat-square&logo=github&color=yellow"/></a>
  <a href="https://github.com/okba14"><img src="https://img.shields.io/github/forks/okba14/Super-RNN?style=flat-square&logo=github&color=orange"/></a>
  <a href="https://github.com/okba14/Super-RNN/issues"><img src="https://img.shields.io/github/issues/okba14/Super-RNN?style=flat-square&color=red"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square"/></a>
  <img src="https://img.shields.io/badge/Made%20with-C%20%26%20❤️-brightgreen?style=flat-square&logo=c&logoColor=white"/>
</p>

---

## 📘 Overview

**Super RNN** is an advanced artificial intelligence system specifically designed for **security analysis on Kali Linux**.  
It implements a **custom Recurrent Neural Network (RNN)** from scratch in **C** to analyze system configurations, logs, and security patterns.  
The system learns from Linux environments, predicts security threats, and generates comprehensive security reports.

---

## 🚀 Key Features

- 🧠 **Custom RNN Implementation:** Built entirely in C with no external dependencies  
- ⚙️ **Adam Optimizer:** Momentum + adaptive learning rate with bias correction  
- 🔐 **Security Analysis:** Deep inspection of logs, configs, and system behavior  
- 📊 **Threat Prediction:** Learns Linux patterns to estimate threat levels  
- 🧾 **Automated Reporting:** Generates detailed reports with threat metrics  
- 💻 **Command Learning:** Understands and executes Linux commands  
- ⏱️ **Real-time Monitoring:** Tracks CPU, memory, and disk activity  
- 🧩 **Early Stopping:** Prevents overfitting dynamically  
- 📈 **Data Normalization:** Stabilizes training & improves accuracy  

---

## 🏗️ Technical Architecture

### 🧩 RNN Structure
- Input-to-hidden weights (`wxh`)  
- Hidden-to-hidden weights (`whh`)  
- Hidden-to-output weights (`why`)  
- Biases (`bxh`, `by`)  
- Hidden state (`h`) and prediction (`y_pred`) buffers  

### ⚙️ Adam Optimizer
- Maintains momentum (`m`) & velocity (`v`)  
- Bias correction for stability  
- Adaptive learning rates (ε smoothing)  

### 🧮 Data Processing
- Text-to-numeric log conversion  
- Normalization/denormalization  
- Gradient clipping (anti-explosion)  

---

## 🛡️ Security Analysis

- Config file analysis (`/etc/passwd`, `/etc/group`, etc.)  
- Log parsing (`/var/log/syslog`, `/var/log/auth.log`, etc.)  
- Command learning from Linux utilities  
- Threat prediction & report generation  

---

## 🧠 Training Process

### 🔧 Initialization
- Xavier weight initialization  
- Adam parameters reset  
- Memory allocation for RNN states  

### 🔁 Training Loop
- Forward pass with dropout  
- Loss via MSE  
- BPTT (Backpropagation Through Time)  
- Adam updates  
- Early stopping (validation loss)  

---

## 🧰 Installation & Compilation

### 🧩 Prerequisites
- 🐧 Linux OS (tested on Kali Linux)  
- 🧱 GCC compiler  
- 📚 Standard C libs (`math.h`, `stdlib.h`, etc.)

### ⚡ Compilation

```bash
gcc -O3 -Wall -Wextra -o super_rnn super_rnn.c -lm
```

# ▶️ Execution

```bash
./super_rnn
sudo ./super_rnn
```

---


# 🧾 Usage

## 1️⃣ System Initialization

* Trains RNN with sample sequence (1→2→3→...)

### 2️⃣ System Learning

* Reads /etc files

* Parses /var/log

* Learns Linux commands

* Monitors CPU/memory

### 3️⃣ Security Analysis

* Detects anomalies

* Predicts threat levels

* Generates /tmp/security_report.txt

# 🧪 Example Output

```bash
=== LEGENDARY KALI LINUX AI SYSTEM ===
Initializing advanced RNN model...
Training initial model...
Epoch 0 | Loss = 0.248321 | Best: 0.248321
...
Security threat level: 0.32
System appears secure.
Security report saved to /tmp/security_report.txt
```

## 📊 Security Report Sample

### Includes:

* System info (CPU cores, memory usage)

* Files analyzed

* Commands learned

* Logs processed

# Threat level: 🟢 LOW / 🟠 MEDIUM / 🔴 HIGH

---


---

## ⚙️ Dependencies

<p align="center">
  <img src="https://img.shields.io/badge/Language-C-blue?style=for-the-badge&logo=c&logoColor=white"/>
  <img src="https://img.shields.io/badge/Platform-Linux-darkgreen?style=for-the-badge&logo=linux&logoColor=white"/>
  <img src="https://img.shields.io/badge/Security-AI%20Enhanced-critical?style=for-the-badge"/>
</p>

| Type | Description |
|------|--------------|
| 🧩 **Standard C Libraries** | `math.h`, `stdio.h`, `stdlib.h`, `time.h`, `float.h`, `string.h`, `dirent.h`, `sys/stat.h`, `sys/types.h`, `unistd.h`, `fcntl.h`, `sys/wait.h` |
| 🧠 **Linux System Files** | `/proc/cpuinfo`, `/proc/meminfo`, `/etc`, `/var/log` |

---

## 👤 Author

<p align="center">
  <img src="https://img.shields.io/badge/Author-GUIAR%20OQBA-blueviolet?style=for-the-badge&logo=github&logoColor=white"/>
  <img src="https://img.shields.io/badge/Nationality-Algeria-00aa00?style=for-the-badge&logo=flag&logoColor=white"/>
</p>

| 🔗 Field | Information |
|----------|--------------|
| 🧑‍💻 **Name** | **GUIAR OQBA** |
| 📧 **Email** | [techokba@gmail.com](mailto:techokba@gmail.com) |
| 🌐 **ORCID** | [0009-0008-1629-0002](https://orcid.org/0009-0008-1629-0002) |
| 💼 **LinkedIn** | [guiar-oqba](https://linkedin.com/in/guiar-oqba) |
| 💻 **GitHub** | [okba14](https://github.com/okba14) |
| 📚 **Zenodo** | [15786076](https://zenodo.org/records/15786076) |
| 📝 **Hashnode** | [@okba](https://hashnode.com/@okba) |
| ✈️ **Telegram** | [@okba_elkantara](https://t.me/okba_elkantara) |
| 🇩🇿 **Made in** | **Algeria** |

---

## 🕓 Version History

📄 See **[CHANGELOG.md](CHANGELOG.md)** for detailed release notes.  
🧩 Current version: `v1.0.0` (Stable Build)

<p align="center">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Build-Stable-success?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Last_Update-2025-orange?style=for-the-badge"/>
</p>

---

## 📜 License

<p align="center">
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-0A66C2?style=for-the-badge&logo=github&logoColor=white"/>
  </a>
  <img src="https://img.shields.io/badge/Copyright-©%202025%20GUIAR%20OQBA-2F4F4F?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Made%20with-💻%20%26%20❤️%20from%20Algeria-006400?style=for-the-badge&logo=linux&logoColor=white"/>
</p>

---

<p align="center">
  <sub><b>Super RNN — Advanced AI System for Kali Linux Security Analysis</b></sub><br>
  <sub>Developed by <b>GUIAR OQBA</b> 🇩🇿 | Empowering AI-based Security on Linux</sub>
</p>
