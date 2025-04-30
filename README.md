## ✒️ Author
- Kevin Sean Hans Lopulalan  
- [Gerend Christopher](https://www.linkedin.com/in/gerendchristopher?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BLAHEs2D5TRWTFm9d8xl7Eg%3D%3D)

## 📚 Overview
The use of computational intelligence in financial markets is on the rise. This project applies Deep Learning, specifically Convolutional Neural Networks (CNNs), to analyze stock market data and assist in algorithmic trading decisions.
While the approach is generic and adaptable to any stock data, this repository demonstrates its application using Apple Inc. (AAPL) as a primary case study.

## 🛠️ Methodology
Input: Time-series stock data (from any stock)
Feature Extraction: 14 technical indicators
Image Construction:
- Each indicator is represented by 16 days of sequential values
- Forming a 16×16 pixel grayscale image
Labels:
- 📈 Buy
- 📉 Sell
- 🤝 Hold
The CNN model learns to recognize trading patterns based on historical stock behavior and technical indicators.

## 🚀 How to Run
1. Clone the repository
   ```bash```
   ```git clone https://github.com/your-username/your-repo-name.git```
   ```cd your-repo-name```
2. Install dependencies
   ```pip install -r requirements.txt```
4. Train and test the model in ```AAPL 18x18.py``` and ```AAPL 16x16.py```.
   PLease note that more detailed information regarding the technical analysis is stored in
   ```src/utils.py```.
5. View the results in the console or saved logs.

## 📄 Report
A detailed project reports (in Indonesian and English) are included in this repository.
You can find it here:
- 📄 Filename: [ENGLISH] Financial Trading with Deep Convolutional Neural Networks.pdf
- 📄 Filename: [INDONESIAN] Financial Trading with Deep Convolutional Neural Networks.pdf

The report explains the background, methods, experiments, and findings of this project in greater depth.

## 🔧 Technologies Used
- Python 3.8+
- TensorFlow / Keras
- NumPy, Pandas
- Scikit-learn, Matplotlib

## 🔑 Keywords
Deep Learning, CNN, Algorithmic Trading, Technical Indicators, Financial Market Prediction, Apple Stock (AAPL)

