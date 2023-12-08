
import requests

# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
url = 'https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency=USD&to_currency=XAU&apikey=66YBSHKLACAC9V5N'
r = requests.get(url)
data = r.json()

print(data)
