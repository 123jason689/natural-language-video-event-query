import gdown
import os

os.makedirs("./models/mobileviclip/weights", exist_ok=True)

gdown.download("https://drive.google.com/uc?id=1BWioaoo8WYYry_Vw72wI-bnpDUzNWRb5", "./models/mobileviclip/weights/mobileviclip_small.pt")
gdown.download("https://drive.google.com/uc?id=1CXUsLE0ShkfSKj0FkwU_3MwlUtNZmJZN", "./models/mobileviclip/weights/mobileclip_s2.pt")
