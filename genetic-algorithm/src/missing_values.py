import pandas as pd
import numpy as np
#MISSING VALUES
full_mat = np.empty((944,1683)) #table for values
full_mat[:] = np.nan

p = pd.read_table('udata.txt')
df = pd.DataFrame(data=p, columns=['user_id','item_id','rating','timestamp']) #get the data frame
means = df.groupby('user_id').agg('mean')
grouped = df.groupby('user_id').agg({"rating":{"size","mean"}})
df=df.drop('timestamp',axis=1)


for i in range(0,len(df)-1):
        cur_item = df.loc[i][1]   #how many movies are blank between records -> sub = next_item - cur_item 
        if (i+1) >len(df): #last item
            next_item = cur_item
        else:
            next_item = df.loc[i+1][1]
            
        user = df.loc[i][0]
        rating = int(round(grouped.loc[user][1])) #replace with mean
        full_mat[user][cur_item]=df.loc[i][2]
        if (next_item - cur_item)>1:  #more than 1 empty 
            j=1
            while(j<=(next_item - cur_item-1)): #for the middle empty lines
                item = j+cur_item
                full_mat[user][item]=rating
                j+=1
                
        elif (next_item - cur_item)<0: #until 1683 movie,   <= for last value
            while(cur_item<1682):
                item = cur_item+1
                full_mat[user][item]=rating
                cur_item+=1
            
            j=1
            while (j<next_item):#empty lines between 1-next_item
                user =df.loc[i+1][0] #next user
                rating = int(round(grouped.loc[user][1])) #next mean rating
                full_mat[user][j]=rating
                j+=1
                
for i in range(1327,1682):
    user,rating=943,3
    full_mat[user][i]=rating

print(full_mat)
pd.DataFrame(full_mat).to_csv("C:\\Users\\lefteris\\.spyder-py3\\ml-100k\\data.csv")
