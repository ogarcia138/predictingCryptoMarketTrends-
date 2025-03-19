import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import requests
import pandas as pd
import threading
import time
import os
import signal
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from pytz import timezone

class BitcoinPriceTracker:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Bitcoin Price Tracker")
        self.root.geometry("900x700")
        self.root.configure(padx=10, pady=10)
        
        # Setup variables
        self.df = None
        self.current_timeframe = "24h"
        self.running = True
        self.update_interval = 5
        
        # Create UI
        self._create_ui()
        self.day_button.state(['pressed'])
        
        # Start update thread
        self.fetch_thread = threading.Thread(target=self._update_loop)
        self.fetch_thread.daemon = True
        self.fetch_thread.start()
    
    def _create_ui(self):
        # Header with title and update time
        header = ttk.Frame(self.root)
        header.pack(fill="x", pady=10)
        ttk.Label(header, text="Bitcoin Price Tracker", font=("Arial", 16, "bold")).pack(side="left")
        self.last_updated_label = ttk.Label(header, text="Last Updated: Never")
        self.last_updated_label.pack(side="right")
        
        # Price information panel
        info = ttk.LabelFrame(self.root, text="Price Information")
        info.pack(fill="x", pady=10)
        
        # Create price info labels
        self.current_price_label = ttk.Label(info, text="Current Price: Loading...", font=("Arial", 12))
        self.high_price_label = ttk.Label(info, text="Period High: Loading...")
        self.low_price_label = ttk.Label(info, text="Period Low: Loading...")
        self.change_label = ttk.Label(info, text="Price Change: Loading...")
        self.change_pct_label = ttk.Label(info, text="Percent Change: Loading...")
        
        # Position labels in grid
        self.current_price_label.grid(row=0, column=0, sticky="w", padx=10, pady=5)
        self.high_price_label.grid(row=1, column=0, sticky="w", padx=10, pady=5)
        self.low_price_label.grid(row=2, column=0, sticky="w", padx=10, pady=5)
        self.change_label.grid(row=0, column=1, sticky="w", padx=10, pady=5)
        self.change_pct_label.grid(row=1, column=1, sticky="w", padx=10, pady=5)
        
        # Timeframe selection
        tf_frame = ttk.Frame(self.root)
        tf_frame.pack(fill="x", pady=10)
        ttk.Label(tf_frame, text="Select Timeframe:").pack(side="left", padx=5)
        
        # Timeframe buttons
        self.hour_button = ttk.Button(tf_frame, text="1 Hour", 
                                      command=lambda: self._change_timeframe("1h"))
        self.day_button = ttk.Button(tf_frame, text="24 Hours", 
                                     command=lambda: self._change_timeframe("24h"))
        self.week_button = ttk.Button(tf_frame, text="1 Week", 
                                      command=lambda: self._change_timeframe("1w"))
        
        self.hour_button.pack(side="left", padx=5)
        self.day_button.pack(side="left", padx=5)
        self.week_button.pack(side="left", padx=5)
        
        # Chart area
        self.chart_frame = ttk.Frame(self.root)
        self.chart_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(8, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.chart_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        self.figure.set_tight_layout(True)
        
        # Status bar
        self.status_var = tk.StringVar(value="Starting application...")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _fetch_data(self, timeframe="24h"):
        """Fetch Bitcoin price data from Kraken API"""
        # Calculate time period
        end_date = datetime.now()
        
        # Set period and interval based on timeframe
        if timeframe == "1h":
            start_date, interval = end_date - timedelta(hours=1), 1  # 1 min intervals
        elif timeframe == "24h":
            start_date, interval = end_date - timedelta(days=1), 5   # 5 min intervals
        else:  # "1w"
            start_date, interval = end_date - timedelta(days=7), 60  # 1 hour intervals
        
        # Kraken API call
        try:
            response = requests.get(
                "https://api.kraken.com/0/public/OHLC",
                params={
                    "pair": "XBTUSD",
                    "interval": interval,
                    "since": int(start_date.timestamp())
                }
            )
            response.raise_for_status()
            data = response.json()
            
            if data["error"]:
                self.status_var.set(f"API Error: {data['error']}")
                return None
                
            return self._process_data(data["result"]["XXBTZUSD"])
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            return None
    
    def _process_data(self, data):
        """Process the raw data from Kraken API"""
        if not data:
            return None
        
        # Create DataFrame and convert types
        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "vwap", "volume", "count"
        ])
        
        # Convert numeric columns
        for col in ["open", "high", "low", "close", "vwap", "volume"]:
            df[col] = df[col].astype(float)
        
        # Convert timestamp to datetime with timezone
        df["date"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
        df["date_pst"] = df["date"].dt.tz_localize('UTC').dt.tz_convert('US/Pacific')
        
        return df.sort_values("date")
    
    def _format_price(self, price):
        """Format price with k notation"""
        return f"${price/1000:.2f}k" if price >= 1000 else f"${price:.2f}"
    
    def _update_display(self):
        """Update all UI elements with latest data"""
        if self.df is None or self.df.empty:
            self.status_var.set("No data available to display.")
            return
        
        # Update timestamp (PST)
        pst_now = datetime.now(timezone('US/Pacific'))
        self.last_updated_label.config(text=f"Last Updated: {pst_now.strftime('%H:%M:%S')} PST")
        
        # Get price data
        current_price = float(self.df['close'].iloc[-1])
        period_high = self.df['high'].max()
        period_low = self.df['low'].min()
        start_price = float(self.df['open'].iloc[0])
        price_change = current_price - start_price
        price_change_pct = (price_change / start_price) * 100
        
        # Update price labels
        self.current_price_label.config(text=f"Current Price: {self._format_price(current_price)}")
        self.high_price_label.config(text=f"Period High: {self._format_price(period_high)}")
        self.low_price_label.config(text=f"Period Low: {self._format_price(period_low)}")
        
        # Format and color change values
        change_color = "green" if price_change >= 0 else "red"
        change_symbol = "+" if price_change >= 0 else ""
        change_text = f"{change_symbol}{self._format_price(abs(price_change))[1:]}"  # Remove $ prefix
        
        self.change_label.config(
            text=f"Price Change: {change_symbol}${change_text}", 
            foreground=change_color
        )
        self.change_pct_label.config(
            text=f"Percent Change: {change_symbol}{price_change_pct:.2f}%", 
            foreground=change_color
        )
        
        # Update chart
        self._update_chart()
    
    def _update_chart(self):
        """Update the price chart"""
        try:
            if self.df is None or self.df.empty:
                return
            
            # Clear and prepare chart
            self.ax.clear()
            
            # Plot data
            self.ax.plot(self.df["date_pst"], self.df["close"], color="orange", label="Closing Price")
            self.ax.fill_between(self.df["date_pst"], self.df["low"], self.df["high"], 
                                color="orange", alpha=0.2, label="Price Range")
            
            # Set labels
            timeframe_names = {"1h": "Past Hour", "24h": "Past 24 Hours", "1w": "Past Week"}
            self.ax.set_title(f"Bitcoin Price (USD) - {timeframe_names.get(self.current_timeframe)} (PST)")
            self.ax.set_xlabel("Time (PST)")
            self.ax.set_ylabel("Price (USD)")
            self.ax.grid(True, linestyle="--", alpha=0.6)
            self.ax.legend()
            
            # Format y-axis with k notation
            from matplotlib.ticker import FuncFormatter
            self.ax.yaxis.set_major_formatter(FuncFormatter(
                lambda x, pos: f'${x/1000:.1f}k' if x >= 1000 else f'${x:.1f}'
            ))
            
            # Format x-axis based on timeframe
            tz = timezone('US/Pacific')
            if self.current_timeframe == "1w":
                self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d', tz=tz))
            else:
                self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=tz))
                
            # Rotate labels and add current price
            for label in self.ax.get_xticklabels():
                label.set_rotation(45)
                label.set_ha('right')
            
            # Add current price annotation
            current_price = self.df["close"].iloc[-1]
            price_text = f"${current_price/1000:.1f}k" if current_price >= 1000 else f"${current_price:.1f}"
            self.ax.annotate(
                price_text, 
                xy=(self.df["date_pst"].iloc[-1], current_price),
                xytext=(10, 0),
                textcoords="offset points",
                fontweight="bold"
            )
            
            # Update canvas
            self.figure.tight_layout()
            self.canvas.draw_idle()
            
        except Exception as e:
            self.status_var.set(f"Chart error: {str(e)}")
    
    def _update_data(self):
        """Fetch and process new data"""
        try:
            self.df = self._fetch_data(self.current_timeframe)
            if self.df is not None and not self.df.empty:
                self.root.after(0, self._update_display)
            else:
                self.status_var.set("Failed to fetch data. Will retry...")
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
    
    def _update_loop(self):
        """Background update loop"""
        while self.running:
            try:
                self._update_data()
                for _ in range(self.update_interval * 2):
                    if not self.running:
                        break
                    time.sleep(0.5)
            except Exception as e:
                print(f"Thread error: {e}")
                time.sleep(2)
    
    def _change_timeframe(self, timeframe):
        """Switch between different timeframes"""
        self.status_var.set(f"Changing to {timeframe}...")
        
        # Update button states
        buttons = {"1h": self.hour_button, "24h": self.day_button, "1w": self.week_button}
        for tf, button in buttons.items():
            button.state(['pressed' if tf == timeframe else '!pressed'])
        
        # Update data for new timeframe
        self.current_timeframe = timeframe
        self._update_data()
    
    def on_closing(self):
        """Clean shutdown"""
        print("Shutting down...")
        self.running = False
        
        # Clean up resources
        try:
            if hasattr(self, 'fetch_thread') and self.fetch_thread.is_alive():
                self.fetch_thread.join(timeout=2)
            
            plt.close('all')
            if hasattr(self, 'figure'):
                plt.close(self.figure)
            if hasattr(self, 'canvas'):
                self.canvas.get_tk_widget().destroy()
                
            self.root.quit()
            self.root.destroy()
            self.root.after(200, lambda: os._exit(0))
        except Exception as e:
            print(f"Error during cleanup: {e}")
            os._exit(0)

def main():
    # Set up signal handlers
    def signal_handler(sig, frame):
        print(f"Received signal {sig}, exiting...")
        os._exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and run app
    try:
        root = tk.Tk()
        root.minsize(800, 600)
        root.rowconfigure(0, weight=1)
        root.columnconfigure(0, weight=1)
        
        app = BitcoinPriceTracker(root)
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        
        print("Application started")
        root.mainloop()
    except Exception as e:
        print(f"Error: {e}")
        os._exit(1)
    
    os._exit(0)

if __name__ == "__main__":
    main()