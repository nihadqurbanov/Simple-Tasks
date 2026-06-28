import yfinance as yf
import pandas as pd
from datetime import datetime

# URL-dəki period1 və period2 timestamp-ləri
period1 = 1420070400  # 2018-06-28
period2 = 1767312000  # 2026-06-28 (təxminən bu gün)

start_date = datetime.fromtimestamp(period1).strftime('%Y-%m-%d')
end_date   = datetime.fromtimestamp(period2).strftime('%Y-%m-%d')

print(f"Tarix aralığı: {start_date} → {end_date}")
print("GOOG məlumatları yüklənir...")

ticker = yf.Ticker("GOOG")
df = ticker.history(start=start_date, end=end_date, auto_adjust=False)

if df.empty:
    print("Məlumat tapılmadı!")
else:
    # İndeksi sıfırla və tarixi sütun et
    df.index = df.index.tz_localize(None)          # timezone-u sil
    df.index.name = "Date"
    df = df.reset_index()
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    # Yalnız əsas sütunları saxla (Yahoo Finance kimi)
    cols = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    available = [c for c in cols if c in df.columns]
    df = df[available]

    # Ədədləri 2 onluq rəqəmlə formatla
    for col in ["Open", "High", "Low", "Close", "Adj Close"]:
        if col in df.columns:
            df[col] = df[col].round(2)

    output_file = "GOOG_history.csv"
    df.to_csv(output_file, index=False)
    print(f"✓ {len(df)} sətir yazıldı → {output_file}")
    print(df.tail(5).to_string(index=False))