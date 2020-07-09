from csv import reader
from scipy.stats import pearsonr
# Locate the most similar neighbors

def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    neighs =list()
    for ind,train_row in enumerate(train):
        dist = pearsonr(test_row, list(map(int,train_row)))
        distances.append((((list(map(int,test_row))), dist[0])))
        neighs.append((ind,dist[0]))
        distances.sort(key=lambda tup: tup[1])
        neighs.sort(key=lambda tup: tup[1])
    #neighbors = list()
    neighs_ids=list()
    for i in range(num_neighbors):
        #neighbors.append(distances[i][0])
        neighs_ids.append(neighs[i][0])
        
    neighbors= find_neighs(neighs_ids,train)
    return neighbors

def find_neighs(neighs_ids,train):
   neighbors=list()
   for i in neighs_ids:
      neighbors.append(list(map(int,train[i])))
   
   return neighbors

def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:continue
			dataset.append(row)
	return dataset
