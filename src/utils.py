import pandas as pd
import numpy as np
import yfinance as yf
import time
from tqdm.auto import tqdm
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from collections import Counter
from operator import itemgetter
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.utils import get_custom_objects

from matplotlib import pyplot as plt

from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, LeakyReLU
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, Callback
from tensorflow.keras import optimizers
from tensorflow.keras.regularizers import l2, l1, l1_l2
from tensorflow.keras.initializers import RandomUniform, RandomNormal
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers
from functools import *
from sklearn.metrics import f1_score
from tensorflow.keras.metrics import AUC

from IPython.display import SVG
from tensorflow.keras.utils import model_to_dot, plot_model
import os

from matplotlib import pyplot as plt
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "last"
from sklearn.metrics import confusion_matrix, roc_auc_score, cohen_kappa_score
import seaborn as sns

class PreprocessingTechIndicator:
    def __init__(self, ticker_symbol, interval, window_size,seed=2):
        self.ticker_symbol = ticker_symbol
        self.interval = interval
        self.window_size = window_size
        self.seed = seed
        pass

    def get_data(self, col_to_analyzed, csv_path):
        # self.stock_data = yf.download(self.ticker_symbol, start=start, end=end)
        self.stock_data = pd.read_csv(csv_path)
        self.stock_data.reset_index(inplace=True)
        self.stock_data['Date'] = pd.to_datetime(self.stock_data['Date'])
        self.col_analized = col_to_analyzed
        self.col_open = 'Open'
        self.col_high = "High"
        self.col_low = "Low"
        self.col_volume = "Volume"

    def seconds_to_minutes(self,seconds):
        return str(seconds // 60) + " minutes " + str(np.round(seconds % 60)) + " seconds"

    def print_time(self,text, stime):
        seconds = (time.time() - stime)
        print(text, self.seconds_to_minutes(seconds))

    def get_RSI_smooth(self, df, col_name, intervals):
        """
        Momentum indicator
        As per https://www.investopedia.com/terms/r/rsi.asp
        RSI_1 = 100 - (100/ (1 + (avg gain% / avg loss%) ) )
        RSI_2 = 100 - (100/ (1 + (prev_avg_gain*13+avg gain% / prev_avg_loss*13 + avg loss%) ) )

        E.g. if period==6, first RSI starts from 7th index because difference of first row is NA
        http://cns.bu.edu/~gsc/CN710/fincast/Technical%20_indicators/Relative%20Strength%20Index%20(RSI).htm
        https://school.stockcharts.com/doku.php?id=technical_indicators:relative_strength_index_rsi
        Verified!
        """

        print("Calculating RSI")
        stime = time.time()
        prev_rsi = np.inf
        prev_avg_gain = np.inf
        prev_avg_loss = np.inf
        rolling_count = 0

        def calculate_RSI(series, period):
            # nonlocal rolling_count
            nonlocal prev_avg_gain
            nonlocal prev_avg_loss
            nonlocal rolling_count

            # num_gains = (series >= 0).sum()
            # num_losses = (series < 0).sum()
            # sum_gains = series[series >= 0].sum()
            # sum_losses = np.abs(series[series < 0].sum())
            curr_gains = series.where(series >= 0, 0)  # replace 0 where series not > 0
            curr_losses = np.abs(series.where(series < 0, 0))
            avg_gain = curr_gains.sum() / period  # * 100
            avg_loss = curr_losses.sum() / period  # * 100
            rsi = -1

            if rolling_count == 0:
                # first RSI calculation
                rsi = 100 - (100 / (1 + (avg_gain / avg_loss)))
                # print(rolling_count,"rs1=",rs, rsi)
            else:
                # smoothed RSI
                # current gain and loss should be used, not avg_gain & avg_loss
                rsi = 100 - (100 / (1 + ((prev_avg_gain * (period - 1) + curr_gains.iloc[-1]) /
                                        (prev_avg_loss * (period - 1) + curr_losses.iloc[-1]))))
                # print(rolling_count,"rs2=",rs, rsi)

            # df['rsi_'+str(period)+'_own'][period + rolling_count] = rsi
            rolling_count = rolling_count + 1
            prev_avg_gain = avg_gain
            prev_avg_loss = avg_loss
            return rsi

        diff = df[col_name].diff()[1:]  # skip na
        for period in tqdm(intervals):
            df['rsi_' + str(period)] = np.nan
            # df['rsi_'+str(period)+'_own_1'] = np.nan
            rolling_count = 0
            res = diff.rolling(period).apply(calculate_RSI, args=(period,), raw=False)
            df['rsi_' + str(period)][1:] = res

        # df.drop(['diff'], axis = 1, inplace=True)
        self.print_time("Calculation of RSI Done", stime)
        return df

    def calculate_wr(self, high, low, close, interval):
        """
        Calculates the Williams %R for a given set of high, low, and closing prices within a specific interval.

        Args:
            high (pandas.Series): A Series containing high prices.
            low (pandas.Series): A Series containing low prices.
            close (pandas.Series): A Series containing closing prices.
            interval (int): The lookback period for calculating the Williams %R.

        Returns:
            pandas.Series: A Series containing the Williams %R values for each row.
        """

        highest_high = high.rolling(window=interval).max()
        lowest_low = low.rolling(window=interval).min()

        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        wr = wr.fillna(-80)  # Set a reasonable default value for missing data

        return wr

    def get_williamR(self, df, high_col, low_col, close_col, intervals):
        """
        Calculates the Williams %R (William's Percent R) indicator for a given DataFrame.

        Args:
            df (pandas.DataFrame): The DataFrame containing high, low, and close prices.
            high_col (str): The column name for the high prices.
            low_col (str): The column name for the low prices.
            close_col (str): The column name for the closing prices.
            intervals (list): A list of lookback periods (intervals) for calculating the Williams %R.

        Returns:
            pandas.DataFrame: The original DataFrame with additional columns named
                            "wr_" + str(interval) for each interval in 'intervals'.
        """

        print("Calculating WilliamR...")

        for interval in tqdm(intervals):
            df["wr_" + str(interval)] = self.calculate_wr(df[high_col], df[low_col], df[close_col], interval)

        print("Calculation of WilliamR Done.")
        return df
    
    def calculate_mfi(self, high, low, close, volume, interval):
        """
        Calculates the Money Flow Index (MFI) for a given set of high, low, close, and volume data within a specific interval.

        Args:
            high (pandas.Series): A Series containing high prices.
            low (pandas.Series): A Series containing low prices.
            close (pandas.Series): A Series containing closing prices.
            volume (pandas.Series): A Series containing volume data.
            interval (int): The lookback period for calculating the MFI.

        Returns:
            pandas.Series: A Series containing the MFI values for each row.
        """

        typical_price = (high + low + close) / 3
        money_flow_ratio = typical_price * volume

        positive_money_flow = money_flow_ratio[close >= close.shift(1)]
        negative_money_flow = money_flow_ratio[close < close.shift(1)]

        mfi = 100 * positive_money_flow.ewm(alpha=1/interval, min_periods=interval).mean() / (
            positive_money_flow.ewm(alpha=1/interval, min_periods=interval).mean() +
            negative_money_flow.ewm(alpha=1/interval, min_periods=interval).mean()
        )
        mfi = mfi.fillna(0)  # Set a reasonable default value for missing data

        return mfi

    def get_mfi(self, df, high_col, low_col, close_col, volume_col, intervals):
        """
        Calculates the Money Flow Index (MFI) for a given DataFrame.

        Args:
            df (pandas.DataFrame): The DataFrame containing high, low, close, and volume data.
            high_col (str): The column name for the high prices.
            low_col (str): The column name for the low prices.
            close_col (str): The column name for the closing prices.
            volume_col (str): The column name for the volume data.
            intervals (list): A list of lookback periods (intervals) for calculating the MFI.

        Returns:
            pandas.DataFrame: The original DataFrame with additional columns named
                            "mfi_" + str(interval) for each interval in 'intervals'.
        """

        print("Calculating MFI...")

        for interval in tqdm(intervals):
            df["mfi_" + str(interval)] = self.calculate_mfi(df[high_col], df[low_col], df[close_col], df[volume_col], interval)

        print("Calculation of MFI Done.")
        return df
    
    def get_ROC(self, df, close_col, intervals):
        """
        Calculates the Rate of Change (ROC) for a given DataFrame.

        Args:
            df (pandas.DataFrame): The DataFrame containing closing prices.
            close_col (str): The column name for the closing prices.
            intervals (list): A list of lookback periods (intervals) for calculating the ROC.

        Returns:
            pandas.DataFrame: The original DataFrame with additional columns named
                            "roc_" + str(interval) for each interval in 'intervals'.
        """

        print("Calculating ROC...")

        for interval in tqdm(intervals):
            df["roc_" + str(interval)] = df[close_col].pct_change(periods=interval) * 100

        print("Calculation of ROC Done.")
        return df
    
    def calculate_cmf(self, high, low, close, volume, interval):
        """
        Calculates the Chaikin Money Flow (CMF) for a given set of high, low, close, and volume data within a specific interval.

        Args:
            high (pandas.Series): A Series containing high prices.
            low (pandas.Series): A Series containing low prices.
            close (pandas.Series): A Series containing closing prices.
            volume (pandas.Series): A Series containing volume data.
            interval (int): The lookback period for calculating the CMF.

        Returns:
            pandas.Series: A Series containing the CMF values for each row.
        """

        money_flow_multiplier = ((close - low) - (high - close)) / (high - low)
        money_flow_volume = money_flow_multiplier * volume
        cmf = money_flow_volume.rolling(window=interval).mean() / volume.rolling(window=interval).sum()
        cmf = cmf.fillna(0)  # Set a reasonable default value for missing data

        return cmf

    def get_CMF(self, df, high_col, low_col, close_col, volume_col, intervals):
        """
        Calculates the Chaikin Money Flow (CMF) for a given DataFrame.

        Args:
            df (pandas.DataFrame): The DataFrame containing high, low, close, and volume data.
            high_col (str): The column name for the high prices.
            low_col (str): The column name for the low prices.
            close_col (str): The column name for the closing prices.
            volume_col (str): The column name for the volume data.
            intervals (list): A list of lookback periods (intervals) for calculating the CMF.

        Returns:
            pandas.DataFrame: The original DataFrame with additional columns named
                            "cmf_" + str(interval) for each interval in 'intervals'.
        """

        print("Calculating CMF...")

        for interval in tqdm(intervals):
            df["cmf_" + str(interval)] = self.calculate_cmf(df[high_col], df[low_col], df[close_col], df[volume_col], interval)

        print("Calculation of CMF Done.")
        return df
    
    def get_CMO(self, df, col_name, intervals):
        """
        Chande Momentum Oscillator
        As per https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/cmo

        CMO = 100 * ((Sum(ups) - Sum(downs))/ ( (Sum(ups) + Sum(downs) ) )
        range = +100 to -100

        params: df -> dataframe with financial instrument history
                col_name -> column name for which CMO is to be calculated
                intervals -> list of periods for which to calculated

        return: None (adds the result in a column)
        """

        print("Calculating CMO")
        stime = time.time()

        def calculate_CMO(series, period):
            # num_gains = (series >= 0).sum()
            # num_losses = (series < 0).sum()
            sum_gains = series[series >= 0].sum()
            sum_losses = np.abs(series[series < 0].sum())
            cmo = 100 * ((sum_gains - sum_losses) / (sum_gains + sum_losses))
            return np.round(cmo, 3)

        diff = df[col_name].diff()[1:]  # skip na
        for period in tqdm(intervals):
            df['cmo_' + str(period)] = np.nan
            res = diff.rolling(period).apply(calculate_CMO, args=(period,), raw=False)
            df['cmo_' + str(period)][1:] = res

        return df
    
    def get_SMA(self, df, col_name, intervals):
        """
        Calculates the Simple Moving Average (SMA) for a given DataFrame.

        Args:
            df (pandas.DataFrame): The DataFrame containing the column for which to calculate the SMA.
            col_name (str): The name of the column to use for the calculation.
            intervals (list): A list of lookback periods (intervals) for calculating the SMA.

        Returns:
            pandas.DataFrame: The original DataFrame with additional columns named
                            "{col_name}_sma_{interval}" for each interval in 'intervals'.
        """

        print("Calculating SMA...")

        for interval in tqdm(intervals):
            df[col_name + '_sma_' + str(interval)] = df[col_name].rolling(window=interval).mean()

        print("Calculation of SMA Done.")
        return df
    
    def get_EMA(self, df, col_name, intervals):
        """
        Calculates the Exponential Moving Average (EMA) for a given DataFrame.

        Args:
            df (pandas.DataFrame): The DataFrame containing the column for which to calculate the EMA.
            col_name (str): The name of the column to use for the calculation.
            intervals (list): A list of lookback periods (intervals) for calculating the EMA.

        Returns:
            pandas.DataFrame: The original DataFrame with additional columns named
                            "{col_name}_ema_{interval}" for each interval in 'intervals'.
        """

        print("Calculating EMA...")

        for interval in tqdm(intervals):
            df[col_name + '_ema_' + str(interval)] = df[col_name].ewm(alpha=1/interval, min_periods=interval).mean()

        print("Calculation of EMA Done.")
        return df
    
    def get_WMA(self, df, col_name, intervals, hma_step=0):
        """
        Momentum indicator
        """
        stime = time.time()
        if (hma_step == 0):
            # don't show progress for internal WMA calculation for HMA
            print("Calculating WMA")

        def wavg(rolling_prices, period):
            weights = pd.Series(range(1, period + 1))
            return np.multiply(rolling_prices.values, weights.values).sum() / weights.sum()

        temp_col_count_dict = {}
        for i in tqdm(intervals, disable=(hma_step != 0)):
            res = df[col_name].rolling(i).apply(wavg, args=(i,), raw=False)
            # print("interval {} has unique values {}".format(i, res.unique()))
            if hma_step == 0:
                df['wma_' + str(i)] = res
            elif hma_step == 1:
                if 'hma_wma_' + str(i) in temp_col_count_dict.keys():
                    temp_col_count_dict['hma_wma_' + str(i)] = temp_col_count_dict['hma_wma_' + str(i)] + 1
                else:
                    temp_col_count_dict['hma_wma_' + str(i)] = 0
                # after halving the periods and rounding, there may be two intervals with same value e.g.
                # 2.6 & 2.8 both would lead to same value (3) after rounding. So save as diff columns
                df['hma_wma_' + str(i) + '_' + str(temp_col_count_dict['hma_wma_' + str(i)])] = 2 * res
            elif hma_step == 3:
                import re
                expr = r"^hma_[0-9]{1}"
                columns = list(df.columns)
                # print("searching", expr, "in", columns, "res=", list(filter(re.compile(expr).search, columns)))
                df['hma_' + str(len(list(filter(re.compile(expr).search, columns))))] = res

        if hma_step == 0:
            self.print_time("Calculation of WMA Done", stime)
        
        return df
    
    def get_HMA(self,df, col_name, intervals):
        import re
        stime = time.time()
        print("Calculating HMA")
        expr = r"^wma_.*"

        if len(list(filter(re.compile(expr).search, list(df.columns)))) > 0:
            print("WMA calculated already. Proceed with HMA")
        else:
            print("Need WMA first...")
            self.get_WMA(df, col_name, intervals)

        intervals_half = np.round([i / 2 for i in intervals]).astype(int)

        # step 1 = WMA for interval/2
        # this creates cols with prefix 'hma_wma_*'
        self.get_WMA(df, col_name, intervals_half, 1)
        # print("step 1 done", list(df.columns))

        # step 2 = step 1 - WMA
        columns = list(df.columns)
        expr = r"^hma_wma.*"
        hma_wma_cols = list(filter(re.compile(expr).search, columns))
        rest_cols = [x for x in columns if x not in hma_wma_cols]
        expr = r"^wma.*"
        wma_cols = list(filter(re.compile(expr).search, rest_cols))

        df[hma_wma_cols] = df[hma_wma_cols].sub(df[wma_cols].values,
                                                fill_value=0)  # .rename(index=str, columns={"close": "col1", "rsi_6": "col2"})
        # df[0:10].copy().reset_index(drop=True).merge(temp.reset_index(drop=True), left_index=True, right_index=True)

        # step 3 = WMA(step 2, interval = sqrt(n))
        intervals_sqrt = np.round([np.sqrt(i) for i in intervals]).astype(int)
        for i, col in tqdm(enumerate(hma_wma_cols)):
            # print("step 3", col, intervals_sqrt[i])
            self.get_WMA(df, col, [intervals_sqrt[i]], 3)
        df.drop(columns=hma_wma_cols, inplace=True)
        self.print_time("Calculation of HMA Done", stime)
        return df
    

    def get_TRIX(self,df, close_col, intervals):
        """
        Calculates the Triple Exponential Moving Average (TRIX) for a given DataFrame.

        Args:
            df (pandas.DataFrame): The DataFrame containing the closing price column.
            close_col (str): The name of the column containing closing prices.
            intervals (list): A list of lookback periods (intervals) for calculating the TRIX.

        Returns:
            pandas.DataFrame: The original DataFrame with additional columns named
                            "trix_" + str(interval) for each interval in 'intervals'.
        """

        print("Calculating TRIX...")

        for interval in tqdm(intervals):
            # Calculate EMA of closing prices
            ema1 = df[close_col].ewm(alpha=1/interval, min_periods=interval).mean()

            # Calculate EMA of EMA1
            ema2 = ema1.ewm(alpha=1/interval, min_periods=interval).mean()

            # Calculate EMA of EMA2
            ema3 = ema2.ewm(alpha=1/interval, min_periods=interval).mean()

            # Calculate TRIX (rate of change of EMA3)
            df["trix_" + str(interval)] = 100 * (ema3 - ema3.shift(1)) / ema3.shift(1)
            df["trix_" + str(interval)] = df["trix_" + str(interval)].fillna(0)  # Set default for missing data

        print("Calculation of TRIX Done.")
        return df
    
    def get_CCI(self, df, high_col, low_col, close_col, intervals):
        """
        Calculates the Commodity Channel Index (CCI) for a given DataFrame.

        Args:
            df (pandas.DataFrame): The DataFrame containing high, low, and closing price columns.
            high_col (str): The name of the column containing high prices.
            low_col (str): The name of the column containing low prices.
            close_col (str): The name of the column containing closing prices.
            intervals (list): A list of lookback periods (intervals) for calculating the CCI.

        Returns:
            pandas.DataFrame: The original DataFrame with additional columns named
                            "cci_" + str(interval) for each interval in 'intervals'.
        """

        print("Calculating CCI...")

        for interval in tqdm(intervals):
            # Calculate typical price
            typical_price = (df[high_col] + df[low_col] + df[close_col]) / 3

            # Calculate average deviation (AD)
            average_deviation = typical_price.rolling(window=interval).std()
            average_deviation = average_deviation.fillna(average_deviation.mean())  # Handle potential missing data in std calculation

            # Calculate CCI
            df["cci_" + str(interval)] = (typical_price - typical_price.rolling(window=interval).mean()) / (0.015 * average_deviation)
            df["cci_" + str(interval)] = df["cci_" + str(interval)].fillna(-80)  # Set default for missing data

        print("Calculation of CCI Done.")
        return df

    def get_DPO(self,df, close_col, intervals):
        """
        Calculates the Detrended Price Oscillator (DPO) for a given DataFrame.

        Args:
            df (pandas.DataFrame): The DataFrame containing the closing price column.
            close_col (str): The name of the column containing closing prices.
            intervals (list): A list of lookback periods (intervals) for calculating the DPO.

        Returns:
            pandas.DataFrame: The original DataFrame with additional columns named
                            "dpo_" + str(interval) for each interval in 'intervals'.
        """

        print("Calculating DPO...")

        for interval in tqdm(intervals):
            # Calculate DPO using pandas shift function
            # df["dpo_" + str(interval)] = df[close_col] - df[close_col].shift(interval)
            df["dpo_" + str(interval)] = df[close_col].shift(interval // 2 + 1) - df[close_col].rolling(window=interval).mean()
        print("Calculation of DPO Done.")
        return df
    
    def calculate_true_range(self,high, low, close):
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range

    def calculate_dm(self,high, low):
        high_diff = high.diff()
        low_diff = low.diff()
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        return plus_dm, minus_dm

    def calculate_smma(self, series, period):
        smma = series.ewm(alpha=1/period, adjust=False).mean()
        return smma

    def calculate_dmi(self,high, low, close, period):
        tr = self.calculate_true_range(high, low, close)
        plus_dm, minus_dm = self.calculate_dm(high, low)
        
        tr_smma = self.calculate_smma(tr, period)
        plus_dm_smma = self.calculate_smma(pd.Series(plus_dm), period)
        minus_dm_smma = self.calculate_smma(pd.Series(minus_dm), period)
        
        plus_di = 100 * (plus_dm_smma / tr_smma)
        minus_di = 100 * (minus_dm_smma / tr_smma)
        
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
        adx = self.calculate_smma(dx, period)
        
        return adx

    def get_DMI(self,df,high_col,low_col,close_col, intervals):
        """
        Calculate the DMI (Directional Movement Index) for given intervals.
        
        Args:
        df (pd.DataFrame): DataFrame containing 'high', 'low', and 'close' columns.
        intervals (list): List of intervals for which to calculate the DMI.
        
        Returns:
        None: Modifies the input DataFrame in place by adding DMI columns.
        """
        stime = time.time()
        print("Calculating DMI")
        
        for i in tqdm(intervals):
            adx = self.calculate_dmi(df[high_col], df[low_col], df[close_col], i)
            df['dmi_' + str(i)] = adx
        
        drop_columns = ['high_delta', 'um', 'low_delta', 'dm', 'pdm', 'pdm_14_ema', 'pdm_14',
                        'close_-1_s', 'tr', 'tr_14_smma', 'atr_14']
        
        expr1 = r'dx_\d+_ema'
        expr2 = r'adx_\d+_ema'
        import re
        drop_columns.extend(list(filter(re.compile(expr1).search, list(df.columns)[9:])))
        drop_columns.extend(list(filter(re.compile(expr2).search, list(df.columns)[9:])))
        
        df.drop(columns=drop_columns, inplace=True, errors='ignore')
        print("Calculation of DMI done in {:.2f} seconds".format(time.time() - stime))
        return df
    
    def create_labels(self, df, col_name, window_size=11):
        """
        Data is labeled as per the logic in research paper
        Label code : BUY => 1, SELL => 0, HOLD => 2
        
        params :
        df => Dataframe with data
        col_name => name of column which should be used to determine strategy
        
        returns : numpy array with integer codes for labels with
                size = total-(window_size)+1
        """
        
        #     self.log("creating label with original paper strategy")
        row_counter = 0
        total_rows = len(df)
        labels = np.zeros(total_rows)
        print(total_rows)
        labels[:] = np.nan
        print("Calculating labels")
        pbar = tqdm(total=total_rows)
        
        while row_counter < total_rows:
            if row_counter >= window_size - 1:
                window_begin = row_counter - (window_size - 1)
                window_end = row_counter
                window_middle = (window_begin + window_end) // 2
                min_ = np.inf
                min_index = -1
                max_ = -np.inf
                max_index = -1
                for i in range(window_begin, window_end + 1):
                    price = df.iloc[i][col_name]
                    if price < min_:
                        min_ = price
                        min_index = i
                    if price > max_:
                        max_ = price
                        max_index = i
        
                if max_index == window_middle:
                    labels[window_middle] = 0
                elif min_index == window_middle:
                    labels[window_middle] = 1
                else:
                    labels[window_middle] = 2
        
            row_counter = row_counter + 1
            pbar.update(1)
        
        pbar.close()
        return labels
    
    def preprocessing(self):
        # Preprocess Technical Indicators
        self.stock_data = self.get_RSI_smooth(self.stock_data,self.col_analized,self.interval)
        self.stock_data = self.get_williamR(self.stock_data,self.col_high,self.col_low,self.col_analized,self.interval)
        self.stock_data = self.get_mfi(self.stock_data,self.col_high,self.col_low,self.col_analized, self.col_volume,self.interval)
        self.stock_data = self.get_ROC(self.stock_data,self.col_analized,self.interval)
        self.stock_data = self.get_CMF(self.stock_data,self.col_high,self.col_low,self.col_analized,self.col_volume,self.interval)
        self.stock_data = self.get_CMO(self.stock_data,self.col_analized,self.interval)
        self.stock_data = self.get_SMA(self.stock_data,self.col_analized,self.interval)
        self.stock_data = self.get_EMA(self.stock_data,self.col_analized,self.interval)
        self.stock_data = self.get_WMA(self.stock_data,self.col_analized,self.interval)
        self.stock_data = self.get_HMA(self.stock_data,self.col_analized,self.interval)
        self.stock_data = self.get_TRIX(self.stock_data,self.col_analized,self.interval)
        self.stock_data = self.get_CCI(self.stock_data, self.col_high,self.col_low,self.col_analized,self.interval)
        self.stock_data = self.get_DPO(self.stock_data,self.col_analized,self.interval)
        self.stock_data = self.get_DMI(self.stock_data, self.col_high,self.col_low,self.col_analized,self.interval)
        # Labeling
        self.label = self.create_labels(self.stock_data,self.col_analized,self.window_size)
        self.label = self.label.astype(np.int8)
        self.stock_data['Labels'] = self.label
        print("Calculating Technical Indicators and Labelling Finished")
        # self.stock_data = self.stock_data.iloc[20:]

    def show_data(self):
        return self.stock_data

    def drop_nan(self, start_index):
        self.stock_data = self.stock_data.iloc[start_index:]

    def splitting_and_scaling(self, train_size, validation_size, selected_columns, print_output=False):
        # self.list_features = list(self.stock_data.loc[:, selected_columns].columns)
        self.list_features = selected_columns

        # split training and test size
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.stock_data.loc[:, selected_columns].values,
                                                                                self.stock_data['Labels'].values, 
                                                                                train_size=train_size,
                                                                                random_state=self.seed,
                                                                                shuffle=True,
                                                                                stratify=self.stock_data['Labels'].values
                                                                               )
        # # Set training size and validation size
        # if 0.7 * self.x_train.shape[0] < 2500:
        #     train_split = 0.8
        # else:
        #     train_split = 0.7

        # split training and validation
        new_train_size = 1-(validation_size/train_size)

        self.x_train, self.x_cv, self.y_train, self.y_cv = train_test_split(self.x_train, 
                                                                            self.y_train, 
                                                                            train_size=new_train_size,
                                                                            random_state=self.seed, 
                                                                            shuffle=True, 
                                                                            stratify=self.y_train
                                                                           )
        # scaling data 
        mm_scaler = MinMaxScaler(feature_range=(0, 1)) # or StandardScaler?
        self.x_train = mm_scaler.fit_transform(self.x_train)
        self.x_cv = mm_scaler.transform(self.x_cv)
        self.x_test = mm_scaler.transform(self.x_test)
        self.x_main = self.x_train.copy()
        
        if print_output == True:
            # print('train_split =', train_size)
            print('Total number of features:', len(self.list_features))
            print("Shape of x, y train/cv/test: {} {} , {} {} , {} {}".format(self.x_train.shape, self.y_train.shape, 
                                                                              self.x_cv.shape, self.y_cv.shape, 
                                                                              self.x_test.shape, self.y_test.shape))
            
    def feature_selection(self, feature_columns_idx):
        self.x_train = self.x_train[:, feature_columns_idx]
        self.x_cv = self.x_cv[:, feature_columns_idx]
        self.x_test = self.x_test[:, feature_columns_idx]

        print("After feature selection")
        print("Shape of x, y train/cv/test: {} {} , {} {} , {} {}".format(self.x_train.shape,
                                                                          self.y_train.shape,
                                                                          self.x_cv.shape,
                                                                          self.y_cv.shape,
                                                                          self.x_test.shape,
                                                                          self.y_test.shape
                                                                         ))

        _labels, _counts = np.unique(self.y_train, return_counts=True)
        print("percentage of class 0 = {}%, class 1 = {}%, class 2 = {}%".format(_counts[0]/len(self.y_train) * 100,
                                                                                 _counts[1]/len(self.y_train) * 100,
                                                                                 _counts[2]/len(self.y_train) * 100)
            )
    
    def get_sample_weights(self,y):
        """
        calculate the sample weights based on class weights. Used for models with
        imbalanced data and one hot encoding prediction.

        params:
            y: class labels as integers
        """

        y = y.astype(int)  # compute_class_weight needs int labels
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)

        print("real class weights are {}".format(class_weights), np.unique(y))
        print("value_counts", np.unique(y, return_counts=True))
        sample_weights = y.copy().astype(float)
        for i in np.unique(y):
            sample_weights[sample_weights == i] = class_weights[i]  # if i == 2 else 0.8 * class_weights[i]
            # sample_weights = np.where(sample_weights == i, class_weights[int(i)], y_)

        return sample_weights

    def reshape_as_image(self,x, img_width, img_height):
        x_temp = np.zeros((len(x), img_height, img_width))
        for i in range(x.shape[0]):
            # print(type(x), type(x_temp), x.shape)
            x_temp[i] = np.reshape(x[i], (img_height, img_width))

        return x_temp

    def f1_weighted(self,y_true, y_pred):
        y_true_class = tf.math.argmax(y_true, axis=1, output_type=tf.dtypes.int32)
        y_pred_class = tf.math.argmax(y_pred, axis=1, output_type=tf.dtypes.int32)
        conf_mat = tf.math.confusion_matrix(y_true_class, y_pred_class)  # can use conf_mat[0, :], tf.slice()
        # precision = TP/TP+FP, recall = TP/TP+FN
        rows, cols = conf_mat.get_shape()
        size = y_true_class.get_shape()[0]
        precision = tf.constant([0, 0, 0])  # change this to use rows/cols as size
        recall = tf.constant([0, 0, 0])
        class_counts = tf.constant([0, 0, 0])

        def get_precision(i, conf_mat):
            print("prec check", conf_mat, conf_mat[i, i], tf.reduce_sum(conf_mat[:, i]))
            precision[i].assign(conf_mat[i, i] / tf.reduce_sum(conf_mat[:, i]))
            recall[i].assign(conf_mat[i, i] / tf.reduce_sum(conf_mat[i, :]))
            tf.add(i, 1)
            return i, conf_mat, precision, recall

        def tf_count(i):
            elements_equal_to_value = tf.equal(y_true_class, i)
            as_ints = tf.cast(elements_equal_to_value, tf.int32)
            count = tf.reduce_sum(as_ints)
            class_counts[i].assign(count)
            tf.add(i, 1)
            return count

        def condition(i, conf_mat):
            return tf.less(i, 3)

        i = tf.constant(3)
        i, conf_mat = tf.while_loop(condition, get_precision, [i, conf_mat])

        i = tf.constant(3)
        c = lambda i: tf.less(i, 3)
        b = tf_count(i)
        tf.while_loop(c, b, [i])

        weights = tf.math.divide(class_counts, size)
        numerators = tf.math.multiply(tf.math.multiply(precision, recall), tf.constant(2))
        denominators = tf.math.add(precision, recall)
        f1s = tf.math.divide(numerators, denominators)
        weighted_f1 = tf.reduce_sum(f.math.multiply(f1s, weights))
        return weighted_f1

    def f1_metric(self,y_true, y_pred):
        """
        this calculates precision & recall
        """

        def recall(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # mistake: y_pred of 0.3 is also considered 1
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall

        def precision(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision

        precision = precision(y_true, y_pred)
        recall = recall(y_true, y_pred)
        # y_true_class = tf.math.argmax(y_true, axis=1, output_type=tf.dtypes.int32)
        # y_pred_class = tf.math.argmax(y_pred, axis=1, output_type=tf.dtypes.int32)
        # conf_mat = tf.math.confusion_matrix(y_true_class, y_pred_class)
        # tf.Print(conf_mat, [conf_mat], "confusion_matrix")

        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
    
    def data_figuring(self, dim):
        get_custom_objects().update({"f1_metric": self.f1_metric, "f1_weighted": self.f1_weighted})
        np.random.seed(self.seed)
        self.sample_weights = self.get_sample_weights(self.y_train)
        # rand_idx = np.random.randint(0, 1000, 30)

        one_hot_enc = OneHotEncoder(sparse_output=False, categories='auto')  # , categories='auto'
        self.y_train = one_hot_enc.fit_transform(self.y_train.reshape(-1, 1))
        self.y_cv = one_hot_enc.transform(self.y_cv.reshape(-1, 1))
        self.y_test = one_hot_enc.transform(self.y_test.reshape(-1, 1))

        dim = dim
        self.x_train = self.reshape_as_image(self.x_train, dim, dim)
        self.x_cv = self.reshape_as_image(self.x_cv, dim, dim)
        self.x_test = self.reshape_as_image(self.x_test, dim, dim)
        # adding a 1-dim for channels (3)
        self.x_train = np.stack((self.x_train,) * 3, axis=-1)
        self.x_test = np.stack((self.x_test,) * 3, axis=-1)
        self.x_cv = np.stack((self.x_cv,) * 3, axis=-1)
        print("Final Shape of x, y train/cv/test: {} {} , {} {} , {} {}".format(self.x_train.shape, self.y_train.shape, 
                                                                                self.x_cv.shape, self.y_cv.shape,
                                                                                self.x_test.shape, self.y_test.shape
                                                                               )
             )

        fig = plt.figure(figsize=(15, 15))
        columns = rows = 4
        for i in range(1, columns*rows +1):
            index = np.random.randint(len(self.x_train))
            img = self.x_train[index]
            fig.add_subplot(rows, columns, i)
            plt.axis("off")
            plt.title('image_'+str(index)+'_class_'+str(np.argmax(self.y_train[index])), fontsize=10)
            plt.subplots_adjust(wspace=0.2, hspace=0.2)
            plt.imshow(img)
        plt.show()


    def f1_custom(self,y_true, y_pred):
        y_t = np.argmax(y_true, axis=1)
        y_p = np.argmax(y_pred, axis=1)
        f1_score(y_t, y_p, labels=None, average='weighted', sample_weight=None, zero_division='warn')

    def create_model_cnn(self, params):
        model = Sequential()

        print("Training with params {}".format(params))
        # (batch_size, timesteps, data_dim)
        # x_train, y_train = get_data_cnn(df, df.head(1).iloc[0]["timestamp"])[0:2]
        conv2d_layer1 = Conv2D(params["conv2d_layers"]["conv2d_filters_1"],
                               params["conv2d_layers"]["conv2d_kernel_size_1"],
                               strides=params["conv2d_layers"]["conv2d_strides_1"],
                               kernel_regularizer=regularizers.l2(params["conv2d_layers"]["kernel_regularizer_1"]),
                               padding='valid', activation="relu", use_bias=True,
                               kernel_initializer='glorot_uniform',
                               input_shape=(self.x_train[0].shape[0],
                                            self.x_train[0].shape[1], 
                                            self.x_train[0].shape[2]
                                           ))
        model.add(conv2d_layer1)
        
        if params["conv2d_layers"]['conv2d_mp_1'] == 1:
            model.add(MaxPool2D(pool_size=3))
            
        model.add(Dropout(params['conv2d_layers']['conv2d_do_1']))
        
        if params["conv2d_layers"]['layers'] == 'two':
            conv2d_layer2 = Conv2D(params["conv2d_layers"]["conv2d_filters_2"],
                                   params["conv2d_layers"]["conv2d_kernel_size_2"],
                                   strides=params["conv2d_layers"]["conv2d_strides_2"],
                                   kernel_regularizer=regularizers.l2(params["conv2d_layers"]["kernel_regularizer_2"]),
                                   padding='valid',activation="relu", use_bias=True,
                                   kernel_initializer='glorot_uniform'
                                  )
            model.add(conv2d_layer2)
            
            if params["conv2d_layers"]['conv2d_mp_2'] == 1:
                model.add(MaxPool2D(pool_size=3))
                
            model.add(Dropout(params['conv2d_layers']['conv2d_do_2']))

        model.add(Flatten())

        model.add(Dense(params['dense_layers']["dense_nodes_1"], activation='relu'))
        model.add(Dropout(params['dense_layers']['dense_do_1']))

        if params['dense_layers']["layers"] == 'two':
            model.add(Dense(params['dense_layers']["dense_nodes_2"], activation='relu',
                            # kernel_regularizer=params['dense_layers']["kernel_regularizer_1"]
                           ))
            model.add(Dropout(params['dense_layers']['dense_do_2']))

        model.add(Dense(3, activation='softmax'))
        
        if params["optimizer"] == 'rmsprop':
            optimizer = optimizers.RMSprop(lr=params["lr"])
        elif params["optimizer"] == 'sgd':
            optimizer = optimizers.SGD(lr=params["lr"], decay=1e-6, momentum=0.9, nesterov=True)
        elif params["optimizer"] == 'adam':
            optimizer = optimizers.Adam(learning_rate=params["lr"], beta_1=0.9, beta_2=0.999, amsgrad=False)
            
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', self.f1_metric])
        # from keras.utils.vis_utils import plot_model use this too for diagram with plot
        # model.summary(print_fn=lambda x: print(x + '\n'))
        return model

    def check_baseline(self,pred, y_test):
        print("size of test set", len(y_test))
        e = np.equal(pred, y_test)
        print("TP class counts", np.unique(y_test[e], return_counts=True))
        print("True class counts", np.unique(y_test, return_counts=True))
        print("Pred class counts", np.unique(pred, return_counts=True))
        holds = np.unique(y_test, return_counts=True)[1][2]  # number 'hold' predictions
        print("baseline acc:", (holds/len(y_test)*100))


    def modeling(self, params, print_summary=False, plot_history=False):
        self.model = self.create_model_cnn(params)
        # plot_model(self.model, to_file='model.png', show_shapes=True, show_layer_names=False)
        if print_summary == True:
            self.model.summary(show_trainable=True, )
            
        # best_model_path = os.path.join('.', 'best_model_keras')
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                        patience=100, min_delta=0.0001)
        # csv_logger = CSVLogger(os.path.join(OUTPUT_PATH, 'log_training_batch.log'), append=True)
        rlp = ReduceLROnPlateau(monitor='val_loss', factor=0.02, patience=20, verbose=1, mode='min',
                                min_delta=0.001, cooldown=1, min_lr=0.0001)

        mcp = ModelCheckpoint("best_model.keras", monitor='val_f1_metric', verbose=1,
                              save_best_only=True, save_weights_only=False, mode='max')  # val_f1_metric
        
        tf.random.set_seed(self.seed)

        self.history = self.model.fit(self.x_train, self.y_train, 
                                      epochs=params['epochs'],
                                      batch_size=params['batch_size'],
                                      verbose=1,
                                      shuffle=True,
                                      validation_data=(self.x_cv, self.y_cv),
                                      callbacks=[mcp, rlp, es], 
                                      sample_weight=self.sample_weights
                                     )
        
        if plot_history == True:
            plt.figure()
            plt.plot(self.history.history['loss'])
            plt.plot(self.history.history['val_loss'])
            plt.title("Model Loss")
            plt.ylabel("Loss")
            plt.xlabel("Epoch")
            plt.legend(['Train Loss', 'Val Loss'])

            plt.figure()
            plt.plot(self.history.history['f1_metric'])
            plt.plot(self.history.history['val_f1_metric'])
            plt.title("Model F1 Score")
            plt.ylabel("F1 Score")
            plt.xlabel('Epoch')
            plt.legend(['Train F1 Score', 'Val F1 Score'])
            
            plt.show()
        
    def evaluate_model(self):
        test_res = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print("keras evaluate =", test_res)
        pred = self.model.predict(self.x_test)
        pred_classes = np.argmax(pred, axis=1)
        y_test_classes = np.argmax(self.y_test, axis=1)
        self.check_baseline(pred_classes, y_test_classes)
        conf_mat = confusion_matrix(y_test_classes, pred_classes)
        print(conf_mat)
        labels = [0,1,2]
        # ax = sns.heatmap(conf_mat, xticklabels=labels, yticklabels=labels, annot=True)
        # ax.xaxis.set_ticks_position('top')
        f1_weighted = f1_score(y_test_classes, pred_classes, labels=None,
                average='weighted', sample_weight=None)
        print("\nF1 score (weighted)", f1_weighted)
        print("\nF1 score (macro)", f1_score(y_test_classes, pred_classes, labels=None,
                                             average='macro', sample_weight=None))
        print("\nF1 score (micro)", f1_score(y_test_classes, pred_classes, labels=None,
                                             average='micro', sample_weight=None))  # weighted and micro preferred in case of imbalance
        # https://scikit-learn.org/stable/modules/model_evaluation.html#cohen-s-kappa --> supports multiclass; ref: https://stats.stackexchange.com/questions/82162/cohens-kappa-in-plain-english
        print("\ncohen's Kappa", cohen_kappa_score(y_test_classes, pred_classes), end='\n\n')

        prec = []
        for i, row in enumerate(conf_mat):
            prec.append(np.round(row[i]/np.sum(row), 2))
            print("precision of class {} = {}".format(i, prec[i]))
        print("\nprecision avg", sum(prec)/len(prec))