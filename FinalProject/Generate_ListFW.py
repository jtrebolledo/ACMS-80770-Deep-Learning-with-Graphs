## Import libraries
import requests
import pandas as pd
import numpy as np

### Search in the Database of Zeolite Structure

# Request to the url with the list of different Zeolite
url = "https://europe.iza-structure.org/IZA-SC/cif/"
html = requests.get(url).content

# Extract table with All Zeolite Frameworks (FWs) and generate a dataframe
df_list = pd.read_html(html)
df = df_list[-1]

# Generate a list with FW
List_FW_pretreatment = np.array(df["Name"])

# Ensure that we only have FW, the value *.cif corresponding to FW
List_FW = np.array([])
for i in List_FW_pretreatment:
    i_str = str(i)
    if i_str.find(".cif") != -1:
        List_FW = np.append(List_FW, i_str[0:3])
        
# Generate a text file with all Zeolite FW
with open('FW_List.txt', 'w') as f:
    for line in List_FW:
        f.write(f"{line}\n")

