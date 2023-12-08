# process.py
import os
import sys
from dotenv import load_dotenv
from datetime import datetime, timedelta
import time
import pandas as pd
import requests

# Load environment variables from .env into the script's environment
load_dotenv()

# Access environment variables
domain_name = os.getenv("DOMAIN_NAME")
api_key = os.getenv("API_KEY")

class PolygonData(): 
    def __init__(self, domain_name, api_key):
        print(domain_name)
        self.domain_name = domain_name
        self.api_key = api_key
        self.column_mapping = {'o': 'open', 'c': 'close', 'h': 'high', 'l':'low',
                               'v': 'volume', 't': 'timestamp', 'n': 'transactions',
                               'vw': 'vwap'}
        self.column_order = [ 'timestamp', 'time', 'open', 'high', 'low', 'close', 
                             'volume', 'vwap', 'transactions' ]

    def fetch(self, **kwargs):
        """
        Fetches and returns data from the Polygon API based on provided parameters.
        https://api.polygon.io/v2/aggs/ticker/C:XAUUSD/range/1/minute/2023-01-09/2023-01-10?apiKey=gdJXmHZY2Ut9QT8hIUN1GYYCyeRaOlmg"


        Args:
            - ticker (str): The ticker symbol for the stock or asset.
            - range (str): The range of data, e.g., "1/minute", "1/hour", "1/day".
            - start_date (str): The start date in the format "YYYY-MM-DD".
            - end_date (str): The end date in the format "YYYY-MM-DD".

        Returns:
            pd.DataFrame or None: A Pandas DataFrame containing the fetched data, or None if an error occurs.

        Example:
            
            'ticker': "C:XAUUSD",
            'frequency': "1/minute",
            'start_date': "2023-01-09",
            'end_date': "2023-01-09",

            df = fetch(ticker="C:XAUUSD", 
                       frequency="1/minute", 
                       start_date= "2023-01-09",
                       'end_date'= "2023-01-09")
        """
        self.ticker = kwargs.get("ticker", "C:XAUUSD")
        self.frequency = kwargs.get("frequency", "1/minute")
        self.start_date = kwargs.get("start_date", datetime.now().strftime("%Y-%m-%d"))
        self.end_date = kwargs.get("end_date", self.start_date)

        # Construct the API endpoint URL with the provided arguments
        url = f"{self.domain_name}/ticker/{self.ticker}/range/{self.frequency}/{self.start_date}/{self.end_date}?apiKey={self.api_key}"
        # Send an HTTP GET request to the API
        response = requests.get(url)
        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()
            results = data['results']
            df = pd.DataFrame(results)
            df = df.rename(columns=self.column_mapping)
            df['time'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.reindex(columns=self.column_order)

        else:
            print(f'Error: {response.status_code} {url}')
            df = pd.DataFrame()
        return df
    
    def to_feather(self, df, dir):
        file_name = f"{self.ticker}_{self.frequency}_{self.start_date}_{self.end_date}"
        file_name = file_name.translate(str.maketrans(":-/", "   ")).replace(" ", "")
        file_path = f"{dir}/feather/{file_name}.feather"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_feather(file_path)

    def to_csv(self, df, dir):
        file_name = f"{self.ticker}_{self.frequency}_{self.start_date}_{self.end_date}"
        file_name = file_name.translate(str.maketrans(":-/", "   ")).replace(" ", "")
        file_path = f"{dir}/csv/{file_name}.csv"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)

    def batch_download(self, date_intervals, dir):
        for interval in date_intervals:
            print(interval)
            interval_start = interval[0]
            interval_end = interval[1]
            df = self.fetch(start_date=interval_start, end_date=interval_end)
            if len(df) == 0:
                print(f'Warning: df could not be generated for interval {interval}!')
            else:
                self.to_feather(df, dir)
                self.to_csv(df, dir)     
            # Wait for 15 sec
            time.sleep(15)

    def make_date_intervals(self, last_date):
        """
        This function creates consecutive time intervals with continuous dates for 
        downloading stock values. Each interval includes both a starting and an ending date,
        and they follow a sequential pattern, where the start of one interval is the day 
        immediately following the end of the previous interval. Furthermore, each interval 
        spans either 3 weekdays or 5 days if weekends are included.
        """
        dates = []
        end_date = datetime.now().date()
        while end_date >= last_date:  
            start_date = end_date - timedelta(days=2)
            if end_date.weekday() == 5:
                start_date -=  timedelta(days=1)
            if end_date.weekday() == 6:
                start_date -=  timedelta(days=2)
            if start_date.weekday() >= 5:
                start_date -=  timedelta(days=2)
            if end_date == last_date:
                start_date = last_date
            dates.append((start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")))
            end_date = start_date - timedelta(days=1)

        return dates
    
    def concatenate(self, dir):
        # List of CSV files in the directory
        path = os.path.join(dir, 'csv')
        if os.path.exists(path):
            csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
        else:
            print(f'Error: path ({path}) does not exist!')
            return None

        if len(csv_files) == 0:
            print(f'Error: path ({path}) is empty!')
            return None

        concatenated_data = pd.DataFrame()
        for csv_file in csv_files:
            file_path = os.path.join(path, csv_file)
            df = pd.read_csv(file_path)
            concatenated_data = pd.concat([concatenated_data, df], ignore_index=True)

        # Sort the concatenated data by timestamp
        concatenated_data['time'] = pd.to_datetime(concatenated_data['time'])
        concatenated_data.sort_values(by='time', inplace=True)
        concatenated_data = concatenated_data.drop_duplicates()

        start_date = concatenated_data['time'].min().strftime('%Y%m%d')
        end_date = concatenated_data['time'].max().strftime('%Y%m%d')
        output_file = os.path.join(dir,f'data_{start_date}_{end_date}.csv')
        concatenated_data.to_csv(output_file, index=False)

        print(f"Concatenated data len({len(concatenated_data)}) saved to {output_file}")

def get_last_date(dir):
    path = os.path.join(dir, 'csv')
    if os.path.exists(path):
        last_end_strs = [epoch_name.split('_')[-1].replace('.csv', '') for epoch_name 
                           in os.listdir(path) if epoch_name.endswith('.csv')]
        if len(last_end_strs) != 0:
            last_end_date = max([datetime.strptime(date, "%Y%m%d") for date in last_end_strs])
            return last_end_date.date()
    return (datetime.now()-timedelta(years=3)).date()
   

def main():
    dir='./data'  # directory name of storage folder
    polygon_data = PolygonData(domain_name, api_key)

    # last_date_str = "2023-09-21"  # Replace with your date string
    # df = polygon_data.fetch(start_date=last_date_str)
    # polygon_data.to_feather(df, dir='./data')

    last_available_date = get_last_date(dir)
    date_intervals = polygon_data.make_date_intervals(last_available_date)
    print(date_intervals)
    print(last_available_date)
    polygon_data.batch_download(date_intervals, dir)
    polygon_data.concatenate(dir)

if __name__ == "__main__":
    main()