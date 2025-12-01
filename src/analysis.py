# src/analysis.py
import matplotlib.pyplot as plt

def visualize_data(df):
    df['price'].plot()  # Ví dụ: vẽ đồ thị giá
    plt.title("Price Data Visualization")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.show()
