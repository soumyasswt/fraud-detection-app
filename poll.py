from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import os
from app.preprocessing import preprocess_df
from app.model_trainer import train_model
import pandas as pd

DATA_DIR = "data"

class Watcher:
    def __init__(self, directory):
        self.observer = Observer()
        self.directory = directory

    def run(self):
        event_handler = Handler()
        self.observer.schedule(event_handler, self.directory, recursive=False)
        self.observer.start()
        print(f"👀 Watching '{self.directory}' for new files...")
        try:
            while True:
                time.sleep(5)
        except KeyboardInterrupt:
            self.observer.stop()
        self.observer.join()

class Handler(FileSystemEventHandler):
    def on_created(self, event):
        if event.src_path.endswith(".csv"):
            print(f"\n📂 New file detected: {event.src_path}")
            try:
                new_df = pd.read_csv(event.src_path)
                new_df = preprocess_df(new_df)

                all_files = [pd.read_csv(os.path.join(DATA_DIR, f)) for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
                all_data = pd.concat([preprocess_df(df) for df in all_files], ignore_index=True)

                train_model(all_data)
                print("✅ Model retrained with new data.\n")
            except Exception as e:
                print(f"❌ Error processing new file: {e}")

if __name__ == "__main__":
    Watcher(DATA_DIR).run()
