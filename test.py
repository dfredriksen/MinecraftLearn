from PIL import Image
import numpy as np

im = Image.open("C:\\Users\\dfred\\AppData\\Roaming\\.minecraft\\screenshots\\2020-04-30_06.44.18.png")
av = np.array(im).mean()

print(av)