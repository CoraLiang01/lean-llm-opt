import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
school_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/school_capacity.csv'
school_capacity_df = pd.read_csv(school_capacity_path, dtype=str, keep_default_na=False)
school_capacity_df['Capacity'] = school_capacity_df['Capacity'].astype(int)
school_capacity_df['School'] = school_capacity_df['School'].str.strip()
schools = list(school_capacity_df['School'])
school_capacities = dict(zip(school_capacity_df['School'], school_capacity_df['Capacity']))
neigh_pop_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/neighborhoods_population.csv'
neigh_pop_df = pd.read_csv(neigh_pop_path, dtype=str, keep_default_na=False)
neigh_pop_df['Neighborhood'] = neigh_pop_df['Neighborhood'].str.strip()
neigh_pop_df['Population_White'] = neigh_pop_df['Population_White'].astype(int)
neigh_pop_df['Population_NonWhite'] = neigh_pop_df['Population_NonWhite'].astype(int)
neighborhoods = list(neigh_pop_df['Neighborhood'])
groups = ['White', 'NonWhite']
pop = {}
for _, row in neigh_pop_df.iterrows():
    n = row['Neighborhood']
    pop[n] = {'White': row['Population_White'], 'NonWhite': row['Population_NonWhite']}
distance_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/distance.csv'
distance_df = pd.read_csv(distance_path, dtype=str, keep_default_na=False)
distance_df['School'] = distance_df['School'].str.strip()
distance_neigh_cols = [col for col in distance_df.columns if col != 'School']
for col in distance_neigh_cols:
    distance_df[col] = distance_df[col].astype(float)
dist = {}
for _, row in distance_df.iterrows():
    s = row['School']
    dist[s] = {}
    for n in neighborhoods:
        dist[s][n] = float(row[n])
district_white_ratio = 0.6
district_nonwhite_ratio = 0.4
tolerance = 0.1
m = Model('School_Busing')
m.Params.OutputFlag = 0
x_vars = m.addVars(schools, neighborhoods, groups, vtype=GRB.INTEGER, lb=0, name='')
for n in neighborhoods:
    for g in groups:
        m.addConstr(quicksum((x_vars[s, n, g] for s in schools)) == pop[n][g])
for s in schools:
    m.addConstr(quicksum((x_vars[s, n, g] for n in neighborhoods for g in groups)) <= school_capacities[s])
for s in schools:
    total_white = quicksum((x_vars[s, n, 'White'] for n in neighborhoods))
    total_nonwhite = quicksum((x_vars[s, n, 'NonWhite'] for n in neighborhoods))
    total_students = total_white + total_nonwhite
    m.addConstr(total_white >= 0.5 * (total_white + total_nonwhite))
    m.addConstr(total_white <= 0.7 * (total_white + total_nonwhite))
m.setObjective(quicksum((x_vars[s, n, g] * dist[s][n] for s in schools for n in neighborhoods for g in groups)), GRB.MINIMIZE)
m.optimize()