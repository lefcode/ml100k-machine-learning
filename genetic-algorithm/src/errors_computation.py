from math import sqrt
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse

def test_dict(user_id,df):
   test_dict={}
   test_range=int(str(user_id)+ str(0))
   i=test_range-11
   while i<test_range-1:
      test_dict.update({df.loc[i][1] : df.loc[i][2]})
      i+=1
   return test_dict

def compute_errors(best,test_dict):
    test_positions=[i for i in test_dict.keys()]
    real_vals=[i for i in test_dict.values()]
    ind_vals=list()

    for i,ind in enumerate(best,1):
        if i in test_positions: ind_vals.append(ind)

    mae_error = mae(ind_vals,real_vals)
    rmse_error = sqrt(mse(ind_vals,real_vals))
    return rmse_error,mae_error