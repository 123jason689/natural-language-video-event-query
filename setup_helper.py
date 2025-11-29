import gdown
import os

os.makedirs("./models/mobileviclip/weights", exist_ok=True)

gdown.download("https://drive.google.com/file/d/1BWioaoo8WYYry_Vw72wI-bnpDUzNWRb5/view?usp=drive_link", "./models/mobileviclip/weights/mobileviclip_small.pt")