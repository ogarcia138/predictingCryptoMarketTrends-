import os
import sys
import tkinter as tk
import signal

# Add the parent directory to sys.path to import btcPT
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from btcPriceTracker.btcPT import BitcoinPriceTracker

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
        root.title("Bitcoin 24h Analysis")
        root.minsize(1200, 1000)
        root.rowconfigure(0, weight=1)
        root.columnconfigure(0, weight=1)
        
        # Create BitcoinPriceTracker instance with fixed 24h timeframe
        app = BitcoinPriceTracker(root)
        
        # Force 24h timeframe and disable other timeframe buttons
        app._change_timeframe("24h")
        app.hour_button.config(state="disabled")
        app.week_button.config(state="disabled")
        
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        
        print("Bitcoin 24h Analyzer started")
        root.mainloop()
    except Exception as e:
        print(f"Error: {e}")
        os._exit(1)
    
    os._exit(0)

if __name__ == "__main__":
    main()