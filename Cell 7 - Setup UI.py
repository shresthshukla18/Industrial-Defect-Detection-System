# Cell 7 

# ==========================================
#  SETUP STREAMLIT & CLOUDFLARE
# ==========================================
!pip install -q streamlit
!wget -q -nc https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
!chmod +x cloudflared-linux-amd64
print(" Streamlit and Cloudflared installed successfully.")
