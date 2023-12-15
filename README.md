# Altin - AI and Statistical Tool for Forex Intraday Trading

## Overview

Welcome to the Altin repository, an advanced AI and statistical tool designed for forex intraday trading. Altin provides a comprehensive set of features for candle analysis, resistance and support detection, trendline analysis, and a variety of indicators such as moving averages, momentum, and more. The tool is equipped with a powerful pipeline to fetch historical forex data for gold prices on a minute-by-minute basis.

## Features

- **Data Fetching Pipeline:** Altin features a pipeline to fetch historical forex data for gold prices at minute intervals, providing a granular view of market movements.

- **Data Cleaning and Candle Generation:** The repository includes DataLoader scripts for cleaning data and generating candles, ensuring data quality for model training.

- **Candle Analysis:** Analyze and interpret candlestick patterns for informed trading decisions.
  
- **Resistance and Support Detection:** Identify key levels of resistance and support to enhance your trading strategies.

- **Trendline Analysis:** Leverage trendline analysis to understand and predict market trends effectively.

- **Indicators:** Explore a rich set of indicators, including moving averages, momentum, and various other indicators, to gain insights into market dynamics.

- **PyTorch Models:** Altin encompasses a collection of PyTorch models for forecasting, featuring:
  - RNN (Recurrent Neural Network)
  - CNN (Convolutional Neural Network)
  - LSTM (Long Short-Term Memory)
  - Transformer-based models
  
## Getting Started

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/Altin.git
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Explore the Codebase:**
   - Navigate through the various modules to explore candle analysis, indicators, models, and more.

4. **Run Data Fetching Pipeline:**
   - Execute the data fetching pipeline script to obtain historical forex data:
     ```bash
     python data_fetching_pipeline.py
     ```

5. **Run Example Model Training:**
   - Explore example scripts for training models:
     ```bash
     python train_rnn_model.py
     ```

## Contribution Guidelines

Contributions to Altin are welcome! If you'd like to contribute, please follow the [contribution guidelines](TODO.md).

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

Special thanks to the contributors and the open-source community for their valuable contributions and support in making Altin a powerful tool for forex trading.

Happy Trading! 📈✨
