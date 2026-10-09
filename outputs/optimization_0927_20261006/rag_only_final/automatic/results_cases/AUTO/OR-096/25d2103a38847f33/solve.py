import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
school_capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/school_capacity.csv', dtype=str, keep_default_na=False)
school_capacity_df['Capacity'] = school_capacity_df['Capacity'].astype(int)
schools = list(school_capacity_df['School'].str.strip())
capacity = {row['School'].strip(): int(row['Capacity']) for (_, row) in school_capacity_df.iterrows()}
neigh_pop_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/neighborhoods_population.csv', dtype=str, keep_default_na=False)
neigh_pop_df['Population_White'] = neigh_pop_df['Population_White'].astype(int)
neigh_pop_df['Population_NonWhite'] = neigh_pop_df['Population_NonWhite'].astype(int)
neighborhoods = list(neigh_pop_df['Neighborhood'].str.strip())
groups = ['White', 'NonWhite']
pop = {}
for (_, row) in neigh_pop_df.iterrows():
    n = row['Neighborhood'].strip()
    pop[n, 'White'] = int(row['Population_White'])
    pop[n, 'NonWhite'] = int(row['Population_NonWhite'])
distance_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/distance.csv', dtype=str, keep_default_na=False)
distance_df['School'] = distance_df['School'].str.strip()
distance_df = distance_df.set_index('School')
for n in neighborhoods:
    distance_df[n] = distance_df[n].astype(float)
dist = {}
for s in schools:
    for n in neighborhoods:
        dist[s, n] = float(distance_df.loc[s, n])
for s in schools:
    if s not in capacity:
        raise ValueError(f'Missing capacity for school {s}')
    for n in neighborhoods:
        if (s, n) not in dist:
            raise ValueError(f'Missing distance for school {s}, neighborhood {n}')
for n in neighborhoods:
    for g in groups:
        if (n, g) not in pop:
            raise ValueError(f'Missing population for neighborhood {n}, group {g}')
total_white = sum((pop[n, 'White'] for n in neighborhoods))
total_nonwhite = sum((pop[n, 'NonWhite'] for n in neighborhoods))
total_students = total_white + total_nonwhite
district_white_ratio = total_white / total_students if total_students > 0 else 0.0
lower_ratio = 0.5
upper_ratio = 0.7
m = Model('SchoolAssignment')
x_vars = m.addVars(schools, neighborhoods, groups, vtype=GRB.INTEGER, lb=0, name='')
for n in neighborhoods:
    for g in groups:
        m.addConstr(quicksum((x_vars[s, n, g] for s in schools)) == pop[n, g], name='')
for s in schools:
    m.addConstr(quicksum((x_vars[s, n, g] for n in neighborhoods for g in groups)) <= capacity[s], name='')
for s in schools:
    total_white_s = quicksum((x_vars[s, n, 'White'] for n in neighborhoods))
    total_assigned_s = quicksum((x_vars[s, n, g] for n in neighborhoods for g in groups))
    m.addConstr(total_white_s >= lower_ratio * total_assigned_s, name='')
    m.addConstr(total_white_s <= upper_ratio * total_assigned_s, name='')
m.setObjective(quicksum((dist[s, n] * x_vars[s, n, g] for s in schools for n in neighborhoods for g in groups)), GRB.MINIMIZE)
m.optimize()