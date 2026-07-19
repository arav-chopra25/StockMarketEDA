import yfinance as yf
# Test with auto_adjust=False to get Adj Close
data = yf.download("AAPL", start="2024-01-01", end="2024-01-10", progress=False, auto_adjust=False)
print("With auto_adjust=False:")
print("Columns:", list(data.columns))
print("\nFirst row:")
print(data.head(1))
