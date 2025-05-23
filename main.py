# 1) Tell python we want the 'pandas' library
#   Pandas helps reads CSV files and 'as pd' is a shorter nickname we chose
import pandas as pd

 # 2) Tell python we want the ability to draw simple plots
 #  matplotlib is the standard plotting library
import matplotlib.pyplot as plt

# 3) Read the CSV file containing the historical natural gas price
#       read_csv loads the file into a "Dataframe"  so pandas can read it
df = pd.read_csv("Nat_Gas.csv")

# Convert dates column  to real date objects
df["Dates"] = pd.to_datetime(df["Dates"])

# Make the dayes column the index so rows are sorted by date.
df = df.set_index("Dates").sort_index()

#Visual check to make sure everything is working
plt.plot(df.index, df["Prices"])
plt.title("Natural Gas Prices (monthly)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show(block = False)

#PART 2 - Now we are going to forecast the prices for the next year
# NOTE: Take into consideration of seasonal trends

#Gives us the model to forecast
from sklearn.linear_model import LinearRegression
from datetime import timedelta

#   1) t is the number of days since first date i.e. forcasted days
df["t"] = (df.index - df.index[0]).days

#   2)
df["month"] = df.index.month
# Turns month into eleven columns named month_1, month_2
dummies = pd.get_dummies(df["month"], prefix = "month", drop_first = True)

#   3) Combine into a single table (features) and Y (target prices)
X = pd.concat([df["t"], dummies], axis = 1)
Y = df["Prices"]

#Create an empty model object and fitting the model can use trends
model = LinearRegression()
model.fit(X, Y)

#Find the last historical date
last_obs = df.index.max()
one_year = last_obs + timedelta(days = 365)

#Predict the prices

def predict_price(date_str: str) -> float:
    date = pd.to_datetime(date_str)

    if date <= last_obs:
        raise ValueError(
            f"Date is not in the allowed forcast window"
            f"Choose a date AFTER {last_obs.date()}."
        )
    if date > one_year:
            raise ValueError(
                f"Extrapolation capped at one year."
                f"Latest allowable date: {one_year.date()}."
            )
    
    t = (date - df.index[0]).days
    row = {"t": t}
    for m in range (2, 13):
        row[f"month_{m}"] = 1 if date.month == m else 0
    X_new = pd.DataFrame([row])

    return float(model.predict(X_new)[0])


if __name__ == "__main__":
     # 1) Build a date time index of future month end prices
    future_dates = pd.date_range(
         start=last_obs + timedelta(days = 1),
         end = one_year,
         freq = "ME"
    )
    # 2) Header
    print("\nForecast: Natural-Gas Month-End Prices")
    print("=======================================")
    print(f"(based on data up to {last_obs.date()})\n")
    print("{:<12}  {:>10}".format("Date", "Price"))

    # 3) Print forecasted prices
    for d in future_dates:
        date_str = d.strftime("%Y-%m-%d")
        price = predict_price(d)
        print(f"{date_str:<12}  {price:>10.2f}")



