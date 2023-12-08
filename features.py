import torch
import joblib
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.widgets import Button, Slider, TextBox
import matplotlib.dates as mdates
# import matplotlib.patches as patches
from utils import load_config
from datetime import datetime, timedelta
from torch.utils.data import DataLoader
from data_loader import ForexDataLoader
from dataset import StackDataset
import mplfinance as mpf
from scipy.signal import find_peaks, find_peaks_cwt
from scipy.signal import savgol_filter
from scipy.signal import argrelextrema
from sklearn.linear_model import LinearRegression

config = load_config('./config.yml')
data_csv_file_path = config['data_csv_file_name']

# Read dataframe 
df = ForexDataLoader.from_csv(data_csv_file_path)
df.index = pd.to_datetime(df.index) + pd.Timedelta(hours=-5)
original_df = df.asfreq('T')  # Set the frequency to 1 minute

# Load test data
print('Loading data... ')
test_data_path = config['test_data_path']
test_dataset = ForexDataLoader.load(test_data_path)


# load transformer
scaler_function_path = config['scaler_function_path']
transformer = joblib.load(scaler_function_path)
batch_size = config['batch_size'] # size of input columns
scaled_test_dataset = transformer.transform(test_dataset)

print(len(test_dataset))
i = 78 # out of 1045

input_seq, target_seg = test_dataset[i]
scaled_input_seq, scaled_target_seg = scaled_test_dataset[i]

reverted_input_seg = transformer.feature_inverse_transform(scaled_input_seq)
reverted_target_seg = transformer.target_inverse_transform(scaled_target_seg)


input_seq_len = input_seq.shape[0]
target_seq_len = target_seg.shape[0]
# for input, target in test_dataset:
#     print(target.shape)

def plot_before_after_transform():
    fig, axs = plt.subplots(4, 1, figsize=(8, 6), sharex=True)

    axs[0].plot(range(input_seq_len), input_seq[:, 4].numpy(), label=f'Input')
    axs[0].plot(range(input_seq_len, input_seq_len + target_seq_len), target_seg[:, 0].numpy(), label=f'Target')

    axs[1].plot(range(input_seq_len), scaled_input_seq[:, 4].numpy(), label=f'Input')
    axs[1].plot(range(input_seq_len, input_seq_len + target_seq_len), scaled_target_seg[:, 0].numpy(), label=f'Target')

    axs[2].plot(range(input_seq_len), reverted_input_seg[:, 4].numpy(), label=f'Input')
    axs[2].plot(range(input_seq_len, input_seq_len + target_seq_len), reverted_target_seg[:, 0].numpy(), label=f'Target')

    axs[3].plot(range(input_seq_len, input_seq_len + target_seq_len), reverted_target_seg[:, 0].numpy(), label=f'Target')

    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    plt.xlabel('minutes (m)')
    plt.suptitle('Line Plots for Each Data Point')

    # Display the plots
    plt.show()

def plot_min_max():
    max_price, max_index = torch.max(target_seg, dim=0)
    min_price, min_index = torch.min(target_seg, dim=0)

    # Determine the order and calculate the difference
    if max_index < min_index:
        order = f"Sell ({min_price}) -> Buy ({max_price})"
        price_difference =  min_price - max_price
    else:
        order = f"Buy ({max_price}) -> Sell ({min_price})"
        price_difference = max_price - min_price

    # Print the results

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    axs[0].plot(range(target_seq_len), target_seg[:, 0].numpy(), label=f'Target')
    axs[0].plot((min_index, max_index), (min_price, min_price), label=f'min')
    axs[0].plot((min_index, max_index), (max_price, max_price), label=f'max')

    axs[0].legend()
    plt.xlabel('minutes (m)')
    plt.suptitle('Line Plots for Each Data Point')

    # Display the plots
    plt.show()

def rsi_indicator(df, n=14):
    '''
    following code gives error of 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    '''
    change = df['close'].diff(1)
    df['avg_gain'] = change.mask(change < 0, 0.0).rolling(window=n).mean()
    df['avg_loss'] = -change.mask(change > 0, -0.0).rolling(window=n).mean()
    return 100 - (100 / (1 + df['avg_gain'] / df['avg_loss'] ))


def add_derived_features(df):
    df = df.copy()
    df['rsi'] = rsi_indicator(df, n=14)
    df['candle_range'] = df['high'] - df['low']
    df['real_body'] = abs(df['open'] - df['close'])
    df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
    df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['average_range'] = (df['high'] - df['low']).rolling(3).mean()
    df['high_gap'] = (df['high']).rolling(3).max() - (df['high']).rolling(3).min()
    return df

def add_peaks(df):
    # filtered_data = df[df['Date'] <= specified_date]
    # Smoothen the high and low data using Savitzky-Golay filter
    window_length = 5  # Adjust the window length as needed
    df['peak'] = np.nan
    df['valley'] = np.nan
    high_smoothed = savgol_filter(df['high'], window_length, 3)  # Adjust the polynomial order as needed
    low_smoothed = savgol_filter(df['low'], window_length, 3)
    # Find peaks and valleys in the smoothened data
    # peaks_i = find_peaks_cwt(df['high'], 2)
    peaks_i, _ = find_peaks(high_smoothed)
    valleys_i, _ = find_peaks(-1 * low_smoothed)
    for i in peaks_i:
        peak = df.iloc[i]
        df.at[peak.name, 'peak'] = peak['high']
    
    for i in valleys_i:
        valley = df.iloc[i]
        df.at[valley.name, 'valley'] = valley['low']

    return df

def add_engulfings(df):
    # print('adding engulfings...')
    # Initialize lists to store engulfing patterns
    df['bullish_engulfing'] = np.nan
    df['bearish_engulfing'] = np.nan
    # Iterate through DataFrame rows to identify engulfing patterns
    for i in range(1, len(df)):
        prev_candle = df.iloc[i - 1]
        current_candle = df.iloc[i]
        if (
            prev_candle['open'] > prev_candle['close']  # Red (bearish) candle
            and current_candle['open'] < prev_candle['close']  # Green (bullish) candle
            and current_candle['close'] > prev_candle['open']
        ):
            df.at[current_candle.name, 'bullish_engulfing'] = current_candle['close']

        if (
            prev_candle['open'] < prev_candle['close']  # Green (bullish) candle
            and current_candle['open'] > prev_candle['close']  # Red (bearish) candle
            and current_candle['close'] < prev_candle['open']
        ):
            df.at[current_candle.name, 'bearish_engulfing'] = current_candle['close']
    return df

def add_hammers(df):
    # Identify hammer patterns
    hammer_criteria = (df['lower_shadow'] > 2 * df['real_body']) & \
                    (df['upper_shadow'] < 0.2 * df['real_body']) & \
                    (df['lower_shadow'] > 0)
    df['hammer'] = df[hammer_criteria]['low']
    return df

def add_spinning_tops(df):
    df['spinning_top'] = np.nan
    # Identify hammer patterns
    spinntop_criteria = (df['lower_shadow'] >= 10 * df['real_body']) & \
        (df['upper_shadow'] >= 10 * df['real_body']) & \
        (df['lower_shadow'] > 0) & (df['upper_shadow'] > 0)
        
    df['spinning_top'] = df[spinntop_criteria]['low']
    return df

def add_supports(df):
    df['support'] = df.iloc[argrelextrema(df['low'].values, np.less_equal, order=5)[0]]['low']
    return df

def add_resitances(df):
    resistance_criteria = (df['high_gap'] < 0.1 * df['average_range'])
    
    # Add resistance levels to the original DataFrame
    df['resistance_level'] = df[resistance_criteria]['high']
    df['resistance'] = df.iloc[argrelextrema(df['high'].values, np.greater_equal, order=5)[0]]['high']
    return df

def add_resitance_strength(df):
    for resistance_index in df['resistance'].dropna().index:
        resistance = df.at[resistance_index, 'resistance']
        # print('resistance', resistance)
        i = 0  # Initialize i
        threshold = 0.05 * (df['high'].max() - df['low'].min())
        resistance_passes = 0
        resistance_bounces = 0
        while i < len(df['high']):
            # Check if the close price is below the resistance threshold
            if df['high'].iloc[i] < resistance - threshold:
                # Check the next point
                j = i + 1
                while j < len(df['high']) - 1:
                    if df['high'].iloc[j] < resistance - threshold:
                        i += 1
                        break
                    elif df['high'].iloc[j] > resistance + threshold:
                        resistance_passes += 1
                        break  # Exit the loop if pass condition is met
                    # Check if the close price reaches the lower threshold (bounce)
                    elif df['high'].iloc[j+1] < resistance - threshold:
                        resistance_bounces += 1
                        break  # Exit the loop if bounce condition is met
                    # Increment the index for the next iteration
                    j += 1

                # Update i to start from the last value of j
                i = j
            else:
                i += 1  # Move to the next i if the condition is not met
        
        # Store the results for this resistance point
        resistance_strength = resistance_bounces - resistance_passes
        df.loc[resistance_index, 'resistance_strength'] = resistance_strength
        df.loc[resistance_index, 'resistance_bounces'] = resistance_bounces
        df.loc[resistance_index, 'resistance_passes'] = resistance_passes
    df.loc[df['resistance_strength'] < 0, 'resistance'] = np.nan
    # print('resistance_strength', df['resistance_strength'].dropna())
    # print(df['resistance_bounces'].dropna())
    # print(df['resistance_passes'].dropna())
    return df

def add_support_strength(df):
    for support_index in df['support'].dropna().index:
        support = df.at[support_index, 'support']
        i = 0  # Initialize i
        threshold = 0.05 * (df['high'].max() - df['low'].min())
        support_passes = 0
        support_bounces = 0
        # print('support', support)
        while i < len(df['high']):
            # Check if the close price is below the support threshold
            if df['low'].iloc[i] > support + threshold:
                # Check the next point
                j = i + 1
                while j < len(df['low']) - 1:
                    if df['low'].iloc[j] > support + threshold:
                        i += 1
                        break
                    elif df['low'].iloc[j] < support - threshold:
                        support_passes += 1
                        break  # Exit the loop if pass condition is met
                    elif df['low'].iloc[j+1] > support + threshold:
                        support_bounces += 1
                        break  # Exit the loop if bounce condition is met
                    # Increment the index for the next iteration
                    j += 1

                # Update i to start from the last value of j
                i = j
            else:
                i += 1  # Move to the next i if the condition is not met
        
        # Store the results for this support point
        support_strength = support_bounces - support_passes
        df.loc[support_index, 'support_strength'] = support_strength
        df.loc[support_index, 'support_bounces'] = support_bounces
        df.loc[support_index, 'support_passes'] = support_passes
    df.loc[df['support_strength'] < 1, 'support'] = np.nan
    # print(df['support_bounces'].dropna())
    # print(df['support_passes'].dropna())
    # print(df['support_strength'].dropna())
    return df


def determine_buy_sell_signal(df, row):
    def find_nearest_support(row):
        valid_supports = df.loc[(~df['support'].isna()) & 
                                (df['support'] < row['close'].iloc[0]) &
                                 (df.index < row.index[-1]) &
                                 (df['support_strength'] > 0), 'support']
        if valid_supports.empty:
            return np.nan
        nearest_support_idx = (valid_supports - row['close'].iloc[0]).abs().idxmin()
        return df.loc[nearest_support_idx, 'support']
    
    def find_next_support(row):
        valid_supports = df.loc[(~df['support'].isna()) & 
                                (df['support'] > row['close'].iloc[0]) &
                                 (df.index < row.index[-1]) &
                                 (df['support_strength'] > 0), 'support']
        if valid_supports.empty:
            return np.nan
        nearest_support_idx = (valid_supports - row['close'].iloc[0]).abs().idxmin()
        return df.loc[nearest_support_idx, 'support']

    def find_nearest_resistance(row):
        valid_resistances = df.loc[(~df['resistance'].isna()) &
                                   (df['resistance'] > row['close'].iloc[0]) &
                                   (df.index < row.index[-1]) &
                                   (df['resistance_strength'] > 0), 'resistance']
        if valid_resistances.empty:
            return np.nan
        nearest_resistance_idx = (valid_resistances - row['close'].iloc[0]).abs().idxmin()
        return df.loc[nearest_resistance_idx, 'resistance']

    def find_next_resistance(row):
        # print(row)
        valid_resistances = df.loc[(~df['resistance'].isna()) &
                                   (df['resistance'] < row['close'].iloc[0]) &
                                   (df.index < row.index[-1]) &
                                   (df['resistance_strength'] > 0), 'resistance']
        if valid_resistances.empty:
            return np.nan
        nearest_resistance_idx = (valid_resistances - row['close'].iloc[0]).abs().idxmin()
        return df.loc[nearest_resistance_idx, 'resistance']

    nearest_support = find_nearest_support(row)
    nearest_resistance = find_nearest_resistance(row)
    next_support = find_next_support(row)
    next_resistance = find_next_resistance(row)
    if np.isnan(nearest_resistance) and np.isnan(next_support):
        resitance_level = np.nan
    else:
        resitance_level = np.nanmin([nearest_resistance, next_support])

    if np.isnan(nearest_support) and np.isnan(next_resistance):
        support_level = np.nan
    else:
        support_level = np.nanmax([nearest_support, next_resistance])

    # print ('support_level: ', support_level)
    # print ('resitance_level: ', resitance_level)

    if not np.isnan(support_level) and np.isnan(resitance_level):
        price_range = df['high'].max() - df['low'].min()
        distance_to_support = row['close'].iloc[0] - support_level
        if distance_to_support < 0.05 * price_range: 
            return 'cnd#1: buy'
        else: 
            return 'cnd#1: hold buy / close'

    elif np.isnan(support_level) and not np.isnan(resitance_level):
        price_range = df['high'].max() - df['low'].min()
        distance_to_resistance = resitance_level - row['close'].iloc[0]
        if distance_to_resistance < 0.05 * price_range: 
            return 'cnd#2: sell'
        else: 
            return 'cnd#2: hold sell / close'
        
    elif not np.isnan(support_level) and not np.isnan(resitance_level):
        distance_to_support = row['close'].iloc[0] - support_level
        distance_to_resistance = resitance_level - row['close'].iloc[0]
        if distance_to_support < 0.05 * distance_to_resistance: 
            return 'cnd#3: buy'
        elif distance_to_resistance < 0.05 * distance_to_support: 
            return 'cnd#3: sell'
        else: 
            return 'cnd#3: hold / stop'

    else:
        return ''

def fit_resitance_line(df):
    # df['resistances'] = df['resistance_level'].dropna().unique()
    return df
    
def add_harami(df):
    df['harami'] = np.nan
    for i in range(1, len(df)):
        prev_candle = df.iloc[i - 1]
        current_candle = df.iloc[i]
        if (
            current_candle['candle_range'] >= 5 * current_candle['real_body']
            and prev_candle[['open', 'close']].max() >= current_candle['high']
            and prev_candle[['open', 'close']].min() <= current_candle['low']
            ):
            df.at[current_candle.name, 'harami'] = current_candle['close']
    return df

def add_position(df, position_label):
    if position_label not in df.columns:
        df[position_label] = np.nan
        # Add or update the position value at the specified timestamp
    df[position_label].iloc[-1] = df['close'].iloc[-1]
    return df

def connect_buy_close(row):
    if not np.isnan(row['buy_close']):
        return df.loc[:row.name, 'buy_open'].dropna().iloc[-1]  
    else:
        return np.nan


def add_indicators(df):
    df = add_derived_features(df)
    df = add_peaks(df)
    df = add_hammers(df)
    df = add_engulfings(df)
    df = add_spinning_tops(df)
    df = add_harami(df)
    df = add_resitances(df)
    df = add_supports(df)
    df = add_resitance_strength(df)
    df = add_support_strength(df)
    return df

def make_plots(df, axs):
    if not isinstance(axs, np.ndarray):      
        axs = [axs]
    for ax in axs:
        ax.clear()

    df = add_indicators(df)
    
    adps = []
    legends = ['', '']

    if not df['peak'].isna().all():
        adps.append(mpf.make_addplot(df['peak'],ax=axs[0],
                            type='scatter', markersize=18, marker='x', color='g'))
        legends.append('peak')
    
    if not df['valley'].isna().all():
        adps.append(mpf.make_addplot(df['valley'],ax=axs[0],
                            type='scatter', markersize=18, marker='x', color='r'))
        legends.append('valley')

    if not df['hammer'].isna().all():
        adps.append(mpf.make_addplot(df['hammer']- 0.5,ax=axs[0],
                            type='scatter', markersize=18, marker='s', color='k'))
        legends.append('hammer')

    if not df['bullish_engulfing'].isna().all():
        adps.append(mpf.make_addplot(df['bullish_engulfing'] - 1,ax=axs[0],
                            type='scatter', markersize=20, marker='^', color='blue'))
        legends.append('bullish_engulfing')
        
    if not df['bearish_engulfing'].isna().all():
        adps.append(mpf.make_addplot(df['bearish_engulfing'] + 1,ax=axs[0],
                            type='scatter', markersize=20, marker='v', color='blue'))
        legends.append('bearish_engulfing')

    if not df['spinning_top'].isna().all():
        adps.append(mpf.make_addplot(df['spinning_top'] - 1,ax=axs[0],
                            type='scatter', markersize=20, marker='D', color='orange'))
        legends.append('spinning_top')
        
    if not df['harami'].isna().all():
        adps.append(mpf.make_addplot(df['harami'] + 1,ax=axs[0],
                            type='scatter', markersize=20, marker='*', color='y'))
        legends.append('harami')
    
    if not df['resistance_level'].isna().all():
        adps.append(mpf.make_addplot(df['resistance_level'],ax=axs[0],
                            type='scatter', markersize=40, marker='_', color='m'))
        legends.append('resistance_level')
    
    # if 'buy_open' in df.columns and not df['buy_open'].isna().all():
    #     adps.append(mpf.make_addplot(df['buy_open'],ax=axs[0],
    #                         type='scatter', markersize=60, marker='^', color='g'))
    #     legends.append('buy_open')
    
    # if 'buy_close' in df.columns and not df['buy_close'].isna().all():
    #     adps.append(mpf.make_addplot(df['buy_close'],ax=axs[0],
    #                         type='scatter', markersize=60, marker='v', color='g'))
    #     legends.append('buy_close')
    
    # if 'sell_open' in df.columns and not df['sell_open'].isna().all():
    #     adps.append(mpf.make_addplot(df['sell_open'],ax=axs[0],
    #                         type='scatter', markersize=60, marker='v', color='r'))
    #     legends.append('sell_open')

    # if 'sell_close' in df.columns and not df['sell_close'].isna().all():
    #     adps.append(mpf.make_addplot(df['sell_close'],ax=axs[0],
    #                         type='scatter', markersize=60, marker='^', color='r'))
    #     legends.append('sell_close')

    if axs.size > 1:
        adps.append(mpf.make_addplot(df['volume'],ax=axs[1],
                            type='bar', panel=1, color='k'))

    if axs.size > 2 and not df['rsi'].isna().all():
        adps.append(mpf.make_addplot(df['rsi'],ax=axs[2],
                            type='line', panel=1, color='k', width=0.2,))
        axs[2].set_ylim(0, 100)
        axs[2].hlines(y=[30, 70], xmin=0, xmax=len(df.index), 
                      colors='r', linestyle='--', linewidth=0.1)
    
    mpf.plot(df, type='candle', style='yahoo', addplot=adps, ax=axs[0], xrotation=0)
    axs[0].text(0.4, 0.94, f'{df.index.max()}', transform=axs[0].transAxes, fontsize=10,
                verticalalignment='center',
                bbox=dict(facecolor='white', alpha=0.5))
    
    resistances = df['resistance'].dropna()
    resistance_strengths = df['resistance_strength'].dropna()
    resistance_strengths = resistance_strengths + 1
    resistance_strengths[resistance_strengths < 0] = 0.05
    axs[0].hlines(y=resistances, xmin=0, xmax=len(df.index), 
                      colors='r', linestyle='-', alpha = 0.3,
                      linewidth=resistance_strengths/2)
    
    legends.append('resistance')
    # axs[0].hlines(y=resistances[-1], xmin=0, xmax=len(df.index), 
    #                   colors='r', linestyle=(0, (50, 30)), linewidth=0.1)
    # legends.append('latest_resistance')

    supports = df['support'].dropna()
    support_strengths = df['resistance_strength'].dropna()
    support_strengths = support_strengths + 1
    support_strengths[support_strengths < 0] = 0.05

    axs[0].hlines(y=supports, xmin=0, xmax=len(df.index), 
                      colors='g', linestyle='-', alpha = 0.3, 
                      linewidth=support_strengths/2)
    legends.append('support')

    signal = determine_buy_sell_signal(df, df.tail(1))
    if 'buy' in signal:
        face_color = 'green'
    elif 'sell' in signal:
        face_color = 'red'
    else:
        face_color = 'gray'

    axs[0].text(.6, .94, signal, color='black',
             transform=axs[0].transAxes, fontsize=10,
             verticalalignment='center', bbox=dict(facecolor=face_color, alpha=0.3))
    axs[0].legend(legends, loc='lower left')
    
def plot_candlesticks(df):
    weekdays_df = df[df.index.to_series().dt.dayofweek < 5]
    random_non_weekend_date = np.random.choice(weekdays_df.index.date)
    original_end_date = pd.to_datetime(str(random_non_weekend_date) + ' 08:00:00')
    
    # original_end_date = datetime(2023, 1, 3, 8, 00, 00)
    # original_end_date = datetime(2021, 10, 11, 8, 00, 00)
    
    original_start_date = original_end_date - timedelta(hours=8)
    start_date = original_start_date
    end_date = original_end_date

    df_of_day = df[start_date:end_date]

    # Select rows within the date range
    # Define the position and size of the button
    n_axs = 3
    gridspec_kw = {'height_ratios': [4, 1, 1]}
    fig, axs = plt.subplots(n_axs, 1, figsize=(12, 8), gridspec_kw=gridspec_kw)
    previous_button_ax = plt.axes([0.125, 0.9, 0.1, 0.06])
    new_button_ax = plt.axes([0.25, 0.9, 0.04, 0.06])
    buy_button_ax = plt.axes([0.35, 0.9, 0.04, 0.03])
    sell_button_ax = plt.axes([0.5, 0.9, 0.04, 0.03])
    save_button_ax = plt.axes([0.73, 0.9, 0.04, 0.06])
    next_button_ax = plt.axes([0.8, 0.9, 0.1, 0.06])
    lot_slider_ax = plt.axes([0.4, 0.93, 0.12, 0.02], facecolor='lightgoldenrodyellow')
    previous_button = Button(previous_button_ax, 'Previous')
    next_button = Button(next_button_ax, 'Next')
    buy_button = Button(buy_button_ax, 'Buy', hovercolor = 'g')
    sell_button = Button(sell_button_ax, 'Sell', hovercolor = 'r')
    new_button = Button(new_button_ax, 'New')
    save_button = Button(save_button_ax, 'Save')
    lot_slider = Slider(lot_slider_ax, 'Lot', 0.1, 0.5, valinit=0.1, valstep=0.1, valfmt='%0.1f')

    note_element = None
    order_size = 0.1
    order_price = 0
    gain = 0
    make_plots(df=df_of_day, axs=axs)
    frq =  df.index.freq.delta.total_seconds()

    orders = {}
    executed_orders = {}

    def on_previous_button_click(df, frq, axs, order_price, gain):
        nonlocal df_of_day
        start_date = df_of_day.index.min() # - timedelta(seconds=frq)
        end_date = df_of_day.index.max() - timedelta(seconds=frq)
        df_of_day = df_of_day[start_date:end_date]
        make_plots(df_of_day, axs=axs)
        display_total_gain(gain, axs[0])
        plt.draw()

    def on_next_button_click(df, frq, axs, order_price, gain):
        nonlocal df_of_day
        # print('on_next: >>>>> ', order_price, gain)
        next_point = df_of_day.index[-1] + timedelta(seconds=frq)
        next_point_point = df_of_day.index[-1] + timedelta(seconds=frq+1)
        # print(df_of_day.index[-1], next_point)
        # print(len(df_of_day.index))
        # print(df[next_point:next_point_point])

        df_of_day = pd.concat([df_of_day, df[next_point:next_point_point]], join='outer')
        # print(len(df_of_day.index))
        make_plots(df_of_day, axs=axs)
        display_total_gain(gain, axs[0])
        display_executed_orders(executed_orders)
        update_orders(orders)
        plt.draw()
    
    def make_note_element(ax, note_text, textbox_props):
        nonlocal note_element
        if note_element:
            note_element.remove()
        note_element = ax.text(0.55, 1.05, note_text, color='black',
                    transform=ax.transAxes, fontsize=10,
                    verticalalignment='center', bbox=textbox_props)
        plt.draw()
        return note_element

    def display_total_gain(total_gain, ax):
        ax.text(0.7, 1.12, f'${total_gain:.1f}', color='black',
                    transform=ax.transAxes, fontsize=12,
                    verticalalignment='center', 
                    bbox=dict(facecolor='white', edgecolor='none', alpha=1))
    
    def update_orders(orders):
        nonlocal df_of_day, axs
        current_price = df_of_day['close'].iloc[-1]
        for i, order_index in enumerate(orders):
            # print(i, orders[order_index])
            order_timestamp = orders[order_index]['order_timestamp']
            order_status = orders[order_index]['status']
            order_iloc = df_of_day.index.get_loc(order_timestamp)
            # print(order_iloc, '_', order_index, '_', order_timestamp, order_status)
            order_type = orders[order_index]['order_type']
            order_type = orders[order_index]['order_type']
            order_price = orders[order_index]['order_price']
            order_size = orders[order_index]['order_size']
            button_ax_position = [0.33, 0.84 - i * 0.03, 0.05, 0.03]
            textbox_ax_position = [0.13, 0.84 - i * 0.03, 0.2, 0.03]
            if order_type == 'buy':
                move = 100 * order_size * (current_price - order_price - 0.06)
                axs[0].scatter([df_of_day.index.get_loc(order_timestamp)], [order_price], marker='^', color='green')
            elif order_type == 'sell':
                move = 100 * order_size * (order_price - current_price - 0.06)
                axs[0].scatter([df_of_day.index.get_loc(order_timestamp)], [order_price], marker='v', color='red')
            if order_status == 'open':
                orders[order_index]['textbox'].set_val(f"{order_type}: {order_size:.1f}@{order_price:.2f}    ${move:.2f}")
                orders[order_index]['textbox'].ax.set_position(textbox_ax_position)
                orders[order_index]['button'].ax.set_position(button_ax_position)
            else:
                orders[order_index]['button'].label.set_text('closed')
                orders[order_index]['button'].label.set_color('#828282')
                orders[order_index]['button'].color = '#D9DDDC'
                orders[order_index]['button'].hovercolor = '#D9DDDC'
                orders[order_index]['button'].pressedcolor = '#D9DDDC'
                orders[order_index]['button'].alpha = 0.2
                # orders[order_index]['textbox'].set_color('gray')
                # orders[order_index]['textbox'].text.set_backgroundcolor('lightgray')

            plt.draw()

    def display_executed_orders(executed_orders):
        nonlocal df_of_day, axs
        for order_index, order in executed_orders.items():
            if order['status'] == 'close':  
                order_timestamp = df_of_day.index.get_loc(order['order_timestamp'])
                execution_timestamp = df_of_day.index.get_loc(order['execution_timestamp'])
                order_type = order['order_type']
                order_price = order['order_price']
                execution_price = order['execution_price']
                (color, marker) = ('red', '^') if order_type == 'sell' else ('blue', 'v')
                axs[0].plot((order_timestamp, 
                        execution_timestamp), 
                        (order_price, execution_price), lw=0.5, color=color)
                axs[0].scatter([order_timestamp], [order_price], marker=marker, color=color)  
                plt.draw()

    def on_close_button_click(order_index, button):
        nonlocal  df_of_day, orders, gain, note_element, axs
        print('position', order_index)
        order_price = orders[order_index]['order_price']
        order_size = orders[order_index]['order_size']

        if orders[order_index]['order_type'] == 'buy':
            gain += 100 * order_size * (df_of_day['close'].iloc[-1] - order_price - 0.06)
        elif orders[order_index]['order_type'] == 'sell':
            gain += 100 * order_size * (order_price - df_of_day['close'].iloc[-1] - 0.06)

        orders[order_index]['button'].disconnect(orders[order_index]['button'].cid)
        orders[order_index]['status'] = 'close'
        executed_orders[order_index] = orders[order_index]
        executed_orders[order_index]['execution_timestamp'] = df_of_day.index[-1]
        executed_orders[order_index]['execution_price'] = df_of_day['close'].iloc[-1]
        textbox_props = dict(boxstyle='round', facecolor='green', edgecolor='none')
        note_text = 'A buy position is closed'
        note_element = make_note_element(axs[0], note_text, textbox_props)
        update_orders(orders)
        display_executed_orders(executed_orders)
        display_total_gain(gain, axs[0])

    def on_buy_button_click(order_index, axs): 
        nonlocal  df_of_day, orders, note_element, order_size
        if df_of_day.index[-1] in orders:
            note_text = "order is open!"
            textbox_props = dict(boxstyle='round', facecolor='red', edgecolor='none')
            note_element = make_note_element(axs[0], note_text, textbox_props)
            return
        order_type = "buy"
        order_size = lot_slider.val
        order_price = df_of_day['close'].iloc[-1] + 0.1
        # df_of_day = add_position(df_of_day, 'buy_open')
        close_button_ax = plt.axes([0.33, 0.84 - len(orders) * 0.03, 0.05, 0.03])
        close_button = Button(close_button_ax, 'close')
        close_button.cid = close_button.on_clicked(lambda event, pos=order_index: on_close_button_click(pos, close_button))
        textbox_ax = plt.axes([0.13, 0.84 - len(orders) * 0.03, 0.2, 0.03])
        textbox = TextBox(textbox_ax, "", initial=f"{order_type}: {order_size:.1f}@{order_price:.2f}    ${0.0:.2f}")

        orders[order_index] = {'order_timestamp': order_index,
                               'order_price': order_price,
                               'order_size': order_size,
                               'order_type': order_type,
                               'button': close_button,
                               'button_ax': close_button_ax,
                               'textbox': textbox,
                               'textbox_ax': textbox_ax,
                               'status': 'open'}
        update_orders(orders)
        plt.draw()
        note_text = 'buy position opened.'
        textbox_props = dict(boxstyle='round', facecolor='white', edgecolor='none')
        textbox_props['facecolor'] = 'green'
        note_element = make_note_element(axs[0], note_text, textbox_props)
 
    def on_sell_button_click(order_index, axs): 
        nonlocal  df_of_day, orders, executed_orders, note_element, order_size
        if df_of_day.index[-1] in orders:
            note_text = "order is open!"
            textbox_props = dict(boxstyle='round', facecolor='red', edgecolor='none')
            note_element = make_note_element(axs[0], note_text, textbox_props)
            return
        order_type = 'sell'
        order_size = lot_slider.val
        order_price = df_of_day['close'].iloc[-1] - 0.1
        # df_of_day = add_position(df_of_day, 'buy_open')
        close_button_ax = plt.axes([0.33, 0.84 - len(orders) * 0.03, 0.05, 0.03])
        close_button = Button(close_button_ax, 'close')
        close_button.cid = close_button.on_clicked(lambda event, pos=order_index: on_close_button_click(pos, close_button))
        textbox_ax = plt.axes([0.13, 0.84 - len(orders) * 0.03, 0.2, 0.03])
        textbox = TextBox(textbox_ax, "", initial=f"{order_type}: {order_size:.1f}@{order_price:.2f}    ${0.0:.2f}")

        orders[order_index] = {'order_timestamp': order_index,
                               'order_price': order_price,
                               'order_size': order_size,
                               'order_type': order_type,
                               'button': close_button,
                               'button_ax': close_button_ax,
                               'textbox': textbox,
                               'textbox_ax': textbox_ax,
                               'status': 'open'}
        update_orders(orders)
        plt.draw()
        note_text = 'sell position opened.'
        textbox_props = dict(boxstyle='round', facecolor='white', edgecolor='none')
        textbox_props['facecolor'] = 'green'
        note_element = make_note_element(axs[0], note_text, textbox_props)
 
    def close_buy_position(df, ax):
        nonlocal start_date, end_date, axs
        buy_open_index = df['buy_open'].last_valid_index()
        buy_open_value = df.loc[buy_open_index, 'buy_open']
        buy_close_index = df['buy_close'].last_valid_index()
        buy_close_value = df.loc[buy_close_index, 'buy_close']
        ax.plot((df.index.get_loc(buy_open_index), 
                     df.index.get_loc(buy_close_index)), 
                     (buy_open_value, buy_close_value), lw=0.5, color='blue')

    def close_sell_position(df, ax):
        sell_open_index = df['sell_open'].last_valid_index()
        sell_open_value = df.loc[sell_open_index, 'sell_open']
        sell_close_index = df['sell_close'].last_valid_index()
        buy_close_value = df.loc[sell_close_index, 'sell_close']
        ax.plot((df.index.get_loc(sell_open_index), 
                     df.index.get_loc(sell_close_index)), 
                     (sell_open_value, buy_close_value), lw=0.5, color='pink')

    # def on_close_button_click(axs, order_price, order_size):
    #     nonlocal df_of_day, gain, note_element
    #     textbox_props = dict(boxstyle='round', facecolor='white', edgecolor='none')
    #     if buy_flag:
    #         gain += 100 * order_size * (df_of_day['close'].iloc[-1] - order_price - 0.06)
    #         textbox_props['facecolor'] = 'green'
    #         note_text = 'buy position is closed'
    #         df_of_day = add_position(df_of_day, 'buy_close')
    #         close_buy_position(df_of_day, axs[0])
    #         buy_flag = False
    #     elif sell_flag:
    #         gain += 100 * order_size * (order_price - df_of_day['close'].iloc[-1] - 0.06)
    #         textbox_props['facecolor'] = 'green'
    #         note_text = 'sell position is closed'
    #         df_of_day = add_position(df_of_day, 'sell_close')
    #         close_sell_position(df_of_day, axs[0])
    #         sell_flag = False
    #     else:
    #         textbox_props['facecolor'] = 'red'
    #         note_text = 'there is no open position.'

    #     note_element = make_note_element(axs[0], note_text, textbox_props)
    #     display_total_gain(gain, axs[0])
    #     plt.draw()

    def on_save_button_click(df, gain): 
        csv_file = 'gain_data.csv'
        latest_date = df.index.max()
        latest_date = latest_date.date()
        if os.path.exists('gain_data.csv'):
            existing_df = pd.read_csv(csv_file)
            existing_dates = existing_df['Date'].astype(str).tolist()
            if str(latest_date) in existing_dates:
                note_text = "date is already exists in the file."
                textbox_props = dict(boxstyle='round', facecolor='red', edgecolor='none')
                make_note_element(axs[0], note_text, textbox_props)
                return
        else:
            existing_df = pd.DataFrame(columns=['Date', 'Gain'])
        new_data = pd.DataFrame([[latest_date, gain]], columns=['Date', 'Gain'])
        existing_df = pd.concat([existing_df, new_data], ignore_index=True)

        # Write the DataFrame to the CSV file
        existing_df.to_csv(csv_file, index=False)
        note_text = "gain saved."
        textbox_props = dict(boxstyle='round', facecolor='blue', edgecolor='none')
        make_note_element(axs[0], note_text, textbox_props)
                
    def on_new_button_click(event):
        nonlocal df
        df = sampled_df.copy()
        plot_candlesticks(df)
    

    previous_button.on_clicked(lambda event: on_previous_button_click(df, frq, axs, order_price, gain))
    next_button.on_clicked(lambda event: on_next_button_click(df, frq, axs, order_price, gain))
    buy_button.on_clicked(lambda event: on_buy_button_click(df_of_day.index[-1], axs))
    sell_button.on_clicked(lambda event: on_sell_button_click(df_of_day.index[-1], axs))   
    new_button.on_clicked(on_new_button_click)
    save_button.on_clicked((lambda event: on_save_button_click(df_of_day, gain)))

    plt.show()

# print(df.index[1:10])
df = original_df.copy()
sampling_frq = ('1T', '5T', '15T')[1]
sampled_df = df.resample(sampling_frq).agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
})

# plot_before_after_transform()
# plot_min_max()
plot_candlesticks(sampled_df)
