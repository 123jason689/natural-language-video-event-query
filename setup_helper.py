import gdown
import os

os.makedirs("./models/mobileviclip/weights", exist_ok=True)

gdown.download("https://drive.google.com/uc?id=1BWioaoo8WYYry_Vw72wI-bnpDUzNWRb5", "./models/mobileviclip/weights/mobileviclip_small.pt")
