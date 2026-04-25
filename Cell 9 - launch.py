# Cell 9

import subprocess
import time
import re
import socket
import os

# 1. Kill old processes to prevent port conflicts
!pkill -f streamlit
!pkill -f cloudflared
time.sleep(2)

# 2. Verify binary name based on your setup step
binary_name = "./cloudflared-linux-amd64"
if not os.path.exists(binary_name):
    if os.path.exists("./cloudflared"):
        binary_name = "./cloudflared"
    else:
        print(" Error: Cloudflare binary not found. Please re-run your setup cell.")

# 3. Port checker function
def is_port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

# 4. Start Streamlit
print(" Starting Streamlit in the background...")
with open("streamlit_log.txt", "w") as f:
    subprocess.Popen(
        ["streamlit", "run", "app.py", "--server.port", "8501", "--server.headless", "true"],
        stdout=f, stderr=f
    )

# 5. Wait for Streamlit to open port 8501
print(" Waiting for Streamlit to boot (Port 8501)...")
timeout = 30
start_time = time.time()
while not is_port_open(8501):
    if time.time() - start_time > timeout:
        print(" Streamlit failed to start within 30 seconds. Check streamlit_log.txt")
        break
    time.sleep(1)
else:
    print(" Streamlit is LIVE! Starting tunnel...")

    # 6. Start Cloudflare Tunnel
    process = subprocess.Popen(
        [binary_name, "tunnel", "--url", "http://localhost:8501", "--no-autoupdate"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    url = None
    # Read output to find the URL
    for _ in range(100):
        line = process.stdout.readline()
        if not line:
            break

        if "trycloudflare.com" in line:
            match = re.search(r"https://[-a-zA-Z0-9]+\.trycloudflare\.com", line)
            if match:
                url = match.group(0)
                print(f"\n YOUR DASHBOARD URL: {url}\n")

        if "Connected" in line or "Connection established" in line:
            print(" Tunnel Connected Successfully! You can click the link above.")
            break

    if not url:
        print(" Failed to parse Tunnel URL. Try running this cell again.")
