# 🌾 FarmWise AI

### AI-Powered Sustainable Farm Decision & Resource Optimization System

FarmWise AI is an AI and data analytics-based agricultural decision-support platform designed to help users understand crop productivity, resource utilization, climate conditions, farm economics, scenario-based resource changes, and agricultural knowledge.

The system follows the workflow:

> **Predict → Analyze → Optimize → Simulate → Recommend**

FarmWise AI combines machine learning, data analytics, scenario simulation, Retrieval-Augmented Generation (RAG), semantic search, and a large language model into an interactive Streamlit dashboard.

---

## 🎯 Problem Statement

Agriculture involves complex decisions related to crop productivity, resource use, climate conditions and farm economics. Farmers and agricultural stakeholders often need to consider multiple factors such as rainfall, temperature, irrigation, fertilizer, pesticide use, yield and profitability together, making data-driven decision-making difficult.

FarmWise AI addresses this challenge by bringing agricultural data analysis, resource-use assessment, climate analysis, economic analysis, scenario simulation and evidence-grounded agricultural advisory into a single decision-support platform.

The system is designed to help users understand historical patterns, compare scenarios and explore more sustainable resource-management decisions using AI and data analytics.

---

## 🚀 Key Features

### 📈 YieldWise
Evaluates agricultural yield prediction using machine-learning models and temporal validation.

Models evaluated include:

- Mean Baseline
- Linear Regression
- Gradient Boosting
- Random Forest
- Extra Trees

The current implementation treats yield prediction as an experimental component because the evaluated dataset showed limited predictive signal.

---

### 🌱 ResourceWise

Analyzes agricultural resource-use patterns involving:

- Fertilizer
- Pesticide
- Irrigation
- Cultivated area
- Yield
- Production

It provides analytical efficiency indicators and identifies records that may require further investigation.

> ResourceWise indicators are analytical observations and are not intended to provide exact fertilizer or pesticide prescriptions.

---

### 💰 CostWise

Analyzes relationships between:

- Crop yield
- Cultivated area
- Market price
- Estimated production
- Farmer profit
- Revenue-related indicators

The module helps users explore agricultural economic patterns using the available dataset.

---

### 🌦️ ClimateWise

Analyzes:

- Rainfall
- Temperature
- Humidity
- Year
- State
- District
- Season
- Crop

It also identifies dataset-relative climate deviations using statistical thresholds and generates a climate stress score.

> Climate stress classifications are dataset-relative screening indicators and should not be interpreted as biological or agronomic diagnoses.

---

### 🔄 What-If Simulator

The What-If Simulator allows users to compare a baseline agricultural condition with an alternative resource scenario.

Example:

```text
Baseline
Fertilizer: 214.08 kg
Pesticide: 36.37 kg

        ↓

Alternative Scenario
Fertilizer: 180 kg
Pesticide: 25 kg