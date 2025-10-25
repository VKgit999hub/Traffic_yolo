import os
import time
import json
from transmit_compressed import transmit_image  # import your existing function

# --- Configuration ---
WATCH_FOLDER = "images_to_send"
SENT_LOG_FILE = "sent_images.json"
SCAN_INTERVAL = 5  # seconds between scans

def load_sent_log():
    """Load the list of already transmitted images."""
    if os.path.exists(SENT_LOG_FILE):
        try:
            with open(SENT_LOG_FILE, "r") as f:
                return set(json.load(f))
        except Exception:
            return set()
    return set()

def save_sent_log(sent_images):
    """Save transmitted images list."""
    with open(SENT_LOG_FILE, "w") as f:
        json.dump(list(sent_images), f, indent=2)

def watch_and_transmit():
    """Continuously watch the folder and send new images."""
    print(f"Watching folder: {WATCH_FOLDER}")
    sent_images = load_sent_log()

    while True:
        try:
            # List all image files in folder
            files = [f for f in os.listdir(WATCH_FOLDER)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            for filename in files:
                filepath = os.path.join(WATCH_FOLDER, filename)

                if filename not in sent_images:
                    print(f"\n🖼️ New image detected: {filename}")
                    success = transmit_image(filepath)

                    if success:
                        print(f"✅ Transmission successful: {filename}")
                        sent_images.add(filename)
                        save_sent_log(sent_images)
                    else:
                        print(f"❌ Transmission failed: {filename}, will retry later")

            time.sleep(SCAN_INTERVAL)

        except KeyboardInterrupt:
            print("\nStopping watcher.")
            break
        except Exception as e:
            print(f"⚠️ Error: {e}")
            time.sleep(5)  # avoid crash loop

if __name__ == "__main__":
    os.makedirs(WATCH_FOLDER, exist_ok=True)
    watch_and_transmit()