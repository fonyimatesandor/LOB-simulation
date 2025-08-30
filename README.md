# Limit Order Book Analysis & Alpha Research

## Overview

This project constructs a high-frequency Limit Order Book (LOB) simulator from first principles using NASDAQ LOBSTER data to analyze market microstructure. The implementation includes a comprehensive research platform capable of processing Level 10 order book data and extracting sophisticated alpha signals across multiple time horizons.

## Key Features
- High-Performance L10 Order Book Simulator: Engineered with O(log n) efficiency using sorted dictionary data structures
- Microstructure Feature Suite: 15+ features including VWAP imbalance, order book entropy, and Herfindahl-Hirschman Index
- Rigorous Validation Framework: Ensures 99.9% fidelity against official LOBSTER snapshots
- Multiple Time Horizon Analysis: 10ms (UHF), 100ms (HFT), and 500ms (intermediate-term) signals
- Alpha Signal Optimization: Linear combination optimization for maximum correlation with future returns


## Key Results
The analysis demonstrates that:
- Simple order book imbalance metrics show significant predictive power
- L1 Imbalance emerges as one of the most powerful predictors of short-term price movements
- Optimized linear combinations of features achieve higher correlations but may not always translate to better returns
- Strategies show positive performance across multiple time horizons (10ms, 100ms, 500ms)

## Future Work
Potential extensions to this research include:
- Cross-asset validation across multiple liquid instruments
- Machine learning approaches for feature combination
- Real-time implementation for paper trading
- Incorporation of transaction costs and execution optimization
- Adaptation to different volatility regimes

## Conclusion
This project demonstrates that significant predictive signals can be extracted from Level 10 order book data, providing a foundation for systematic alpha research in high-frequency trading. The rigorous implementation of a high-fidelity order book simulator enables continued research into market microstructure and quantitative alpha generation.