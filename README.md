# Walk-Forward Market Regime Detection (S&P 500)

## 1. Project Overview
This project implements a **Hidden Markov Model (HMM)** to detect latent market regimes ("Bull", "Bear", "High Volatility") and dynamically adjust portfolio exposure. Unlike standard backtests, this project uses **Walk-Forward Analysis** to eliminate look-ahead bias, retraining the model iteratively on past data to predict future states.

**Objective:** To minimize Maximum Drawdown and volatility during major market crashes (2008, 2020, 2022).

## 2. Technical Implementation
* **Algorithm:** Gaussian HMM (3 States, Full Covariance).
* **Validation:** Expanding Window Walk-Forward (Retraining every 10 days).
* **Stabilization:** Implemented `min_covar` checks and dynamic state-mapping to handle regime flipping and numerical instability in rolling windows.
* **Stack:** Python, `hmmlearn`, Pandas, Matplotlib.

## 3. Walk-Forward Results (Out-of-Sample 2005–2024)
The strategy functions as a "Volatility Shield," significantly reducing exposure during stress periods.

| Metric | HMM Strategy (Walk-Forward) | Benchmark (SPY) |
| :--- | :--- | :--- |
| **Sharpe Ratio** | **0.73** | 0.52 |
| **Annual Volatility** | **6.43%** | 19.07% |
| **Total Return** | **154.10%** | 632.69% |

### Key Findings
* **Risk Reduction:** The strategy successfully reduced Annual Volatility by **~66%** (from 19.07% to 6.43%), effectively transforming the risk profile of the S&P 500 into that of a high-yield bond fund.
* **Crisis Alpha:** The model successfully shifted to Cash during the **2008 Financial Crisis** and **2020 COVID Crash**, protecting capital when volatility spiked.
* **Defensive Nature:** Due to the lag inherent in rolling-window training, the model acts conservatively, often delaying re-entry during V-shaped recoveries. This highlights the trade-off between drawdown protection (Safety) and total upside capture (Growth).
