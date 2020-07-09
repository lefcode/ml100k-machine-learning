import pandas as pd
import numpy as np
#CENTERING BEGIN
p = pd.read_table('udata.txt')
df = pd.DataFrame(data=p) #get the data frame
grouped = df.groupby('user_id').agg({"rating":{"size","mean"}})
in_df=0 
in_grouped=1
while ((in_df<len(df)) and (in_grouped<=len(grouped))):
    num = grouped.loc[in_grouped][1] #size in df 
    num+=in_df
    i=in_df
    while(i<num):
        df.loc[i][2]-=grouped.loc[in_grouped][0] #rate column - mean column
        i+=1
        
    in_grouped +=1
    in_df=num
np.savetxt(r'C:\\Users\\lefteris\\.spyder-py3\\ml-100k\\centered_data.txt', df.values, fmt='%d')
'''After centering data in rating column: 
   Max value: 3 
   Min value: -3
'''
#CENTERING END