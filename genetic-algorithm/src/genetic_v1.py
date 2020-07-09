# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 19:52:10 2020

@author: lefteris Mantas
"""

import pandas as pd
import numpy as np
from deap import base, creator,tools
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
import random
from matplotlib import pyplot
import errors_computation as e #import file errors_computation.py
import neighborhood as n #import file neighborhood.py
from functools import partial

############################################################################
'''Global values'''
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

data= pd.read_csv('C:\\Users\\lefteris\\.spyder-py3\\Genetic_Algorithm\\data\\data_base.csv')
full_data = n.load_csv('C:\\Users\\lefteris\\.spyder-py3\\Genetic_Algorithm\\data\\data_full.csv')
p = pd.read_table('C:\\Users\\lefteris\\.spyder-py3\\Genetic_Algorithm\\data\\ml-100k\\ua.test')
df = pd.DataFrame(data=p) #get the data frame
print('Data retrieved')

user_id=269

#change input for different experiments
print('Insert the following GA parameters')
'''Here are the best parameters for this genetic algorithm'''
POP_SIZE = int(input('Insert POP_SIZE')) #size of population
P_CROS = float(input('Insert P_CROS')) #probability for crossover 
P_MUT = float(input('Insert P_MUT')) #probability for mutation
MAX_GEN= int(input('Insert MAX_GEN')) #max number of generations
CHRM_LENGTH = 1682 #chromosome length

def setting(user_id):
   global user,neighbors
   user = data.loc[user_id,:] #user chromosome 
   neighbors=n.get_neighbors(full_data, list(map(int,full_data[user_id-1])),10) 
############################################################################

def pop_creator(user):
   p=[int(x) if not np.isnan(x) else random.randint(1,5) for x in user]
   return p

def evaluation_func(individual):
   pears = [pearsonr(individual, neighbor) for neighbor in neighbors]
   pears_scaled = MinMaxScaler().fit_transform(pears)
   pears_av = np.mean(pears_scaled)
   return pears_av,

def repair_function(user,offspring):
    for individual in offspring:
        for i in range(len(individual)):
            if not np.isnan(user[i]):
                individual[i] = user[i]
                
def check_stopping_criterio(gens,maxfit,maxfitvals):
   count=0
   if gens>10:
      for g in range(5):
         if ((maxfit-maxfitvals[-2-g])/maxfitvals[-2-g])< 0.01: 
            count+=1 #not increased by 1%
        
         if count>=5:
            print('Stopping criteria reached')
            return True
            
   return False
############################################################################

def set_toolbox():
   global toolbox
   individual_gen = partial(pop_creator, user)   
   toolbox = base.Toolbox()
   toolbox.register("individualCreator", lambda individual, individual_gen: individual(individual_gen()), creator.Individual,
                    individual_gen)
   toolbox.register("populationCreator", tools.initRepeat,
                    list, toolbox.individualCreator)
   
   toolbox.register("evaluate", evaluation_func)
   
   toolbox.register("select", tools.selTournament, tournsize=3)
   toolbox.register("mate", tools.cxPartialyMatched)
   toolbox.register("mutate", tools.mutUniformInt, low=1, up=5, indpb=0.5)
   
############################################################################
def main(user_id):
    set_toolbox()
    print('Toolbox is set')
    maxfitvals=[]
    rmse_error_list=[]
    mae_error_list=[]
    stop_alg=False

    test_dict = e.test_dict(user_id,df)
    population = toolbox.populationCreator(n=POP_SIZE)
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
        
    fits = [ind.fitness.values[0] for ind in population]
    gens = 0
    '''Create and iterate through generations '''
    while (gens < MAX_GEN) and( max(fits)<CHRM_LENGTH) and (stop_alg==False):
        gens+=1
        print("-- Generation %i --" % gens)
        
        '''Selecting the next generation individuals '''
        offspring = toolbox.select(population, len(population))
        '''Clone the selected individuals'''
        offspring = list(map(toolbox.clone, offspring))
        
        '''Apply crossover on the offspring'''
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < P_CROS:
                toolbox.mate(child1, child2)
                '''Delete the fitness value of child'''
                del child1.fitness.values
                del child2.fitness.values
                
        '''Apply mutation '''
        for mutant in offspring:
            if random.random() < P_MUT:
                toolbox.mutate(mutant)
                del mutant.fitness.values
                
        '''Apply the repair function on the offspring'''
        repair_function(user,offspring)
        
        '''Evaluate the individuals with an invalid fitness '''
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            
        '''Replace population with next generation individual '''   
        population[:] = offspring
        
        '''Get all useful information of each generation'''
        fit_values = [ind.fitness.values[0] for ind in population]
        best = population[np.argmax([toolbox.evaluate(x) for x in population])]
        maxfit=max(fit_values)
        maxfitvals.append(maxfit)

        print('Generation {}: Max Fitness = {}'.format(gens,maxfit))

        rmse_error,mae_error = e.compute_errors(best,test_dict)
        rmse_error_list.append(rmse_error)
        mae_error_list.append(mae_error)
        stop_alg= check_stopping_criterio(gens,maxfit,maxfitvals)
    
    #best = population[np.argmax([toolbox.evaluate(x) for x in population])]
    
    best_fit=max(maxfitvals)
    
    #rmse_error,mae_error = e.compute_errors(best,test_dict)
    '''Plot the maximum fitness values and the errors through the generations'''
    pyplot.plot(maxfitvals,color='green',label='Fitness Value of Best Individual')
    pyplot.plot(rmse_error_list,color='blue',label='RMSE error')
    pyplot.plot(mae_error_list,color='red',label='MAE error')
    pyplot.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    pyplot.xlabel("Generations")
    pyplot.ylabel("Average Fitness of Best")
    pyplot.title("Evolution Curve")
    pyplot.show()

    return best_fit,gens,np.mean(rmse_error_list),np.mean(mae_error_list)
 
def results(user_id):
   averages=[]
   gens_av=[]
   total_rmse_error_list=[]
   total_mae_error_list=[]
   for i in range(1,11): #we run 10 times the GA and take the average values 
      best,gens,rmse_error,mae_error = main(user_id)
      averages.append(best)
      gens_av.append(gens)
      print('rmse:',rmse_error)
      print('mae:',mae_error)
      total_rmse_error_list.append(rmse_error)
      total_mae_error_list.append(mae_error)
      print('Execution number:',i)
   
   TOTAL_AVERAGE=np.mean(averages)
   GENS_AVERAGE=np.mean(gens_av) 
   RMSE_AVERAGE=np.mean(total_rmse_error_list)
   MAE_AVERAGE=np.mean(total_mae_error_list)
   print('BEST AVERAGE & GENS')
   print(TOTAL_AVERAGE,GENS_AVERAGE)
   print('RMSE AVERAGE & MAE AVERAGE')
   print(RMSE_AVERAGE,MAE_AVERAGE)   
   
if __name__ == "__main__":
   setting(user_id)
   results(user_id)