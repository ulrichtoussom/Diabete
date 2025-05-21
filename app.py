import yfinance as yf 

ticket =  'BTC-EUR'
data = None

try:
    data =  yf.download(ticket, start='2024-01-01', end = '2025-01-01', interval='1d') 
except:
    print('Une erreur c\'est produite lors du téléchargement des données')
    
if data is not None:
    data.to_csv(ticket + '.csv')
    print('Données téléchargées avec succès')