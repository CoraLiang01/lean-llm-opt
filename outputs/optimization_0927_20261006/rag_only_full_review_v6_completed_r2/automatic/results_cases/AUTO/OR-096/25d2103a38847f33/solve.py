import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
school_capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/school_capacity.csv', dtype=str, keep_default_na=False)
school_capacity_df['School'] = school_capacity_df['School'].str.strip()
school_capacity_df['Capacity'] = school_capacity_df['Capacity'].astype(int)
schools = list(school_capacity_df['School'])
school_capacities = dict(zip(school_capacity_df['School'], school_capacity_df['Capacity']))
neigh_pop_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/neighborhoods_population.csv', dtype=str, keep_default_na=False)
neigh_pop_df['Neighborhood'] = neigh_pop_df['Neighborhood'].str.strip()
neigh_pop_df['Population_White'] = neigh_pop_df['Population_White'].astype(int)
neigh_pop_df['Population_NonWhite'] = neigh_pop_df['Population_NonWhite'].astype(int)
neighborhoods = list(neigh_pop_df['Neighborhood'])
pop_white = dict(zip(neigh_pop_df['Neighborhood'], neigh_pop_df['Population_White']))
pop_nonwhite = dict(zip(neigh_pop_df['Neighborhood'], neigh_pop_df['Population_NonWhite']))
distance_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/distance.csv', dtype=str, keep_default_na=False)
distance_df['School'] = distance_df['School'].str.strip()
distance_df = distance_df.set_index('School')
for n in neighborhoods:
    distance_df[n] = distance_df[n].astype(float)
distance = {(s, n): float(distance_df.loc[s, n]) for s in schools for n in neighborhoods}
groups = ['White', 'NonWhite']
total_white = sum((pop_white[n] for n in neighborhoods))
total_nonwhite = sum((pop_nonwhite[n] for n in neighborhoods))
total_students = total_white + total_nonwhite
district_white_ratio = total_white / total_students if total_students > 0 else 0.0
min_white_ratio = 0.5
max_white_ratio = 0.7
m = Model('SchoolAssignment')
x_vars = m.addVars(schools, neighborhoods, groups, vtype=GRB.INTEGER, lb=0, name='')
for n in neighborhoods:
    m.addConstr(quicksum((x_vars[s, n, 'White'] for s in schools)) == pop_white[n])
    m.addConstr(quicksum((x_vars[s, n, 'NonWhite'] for s in schools)) == pop_nonwhite[n])
for s in schools:
    m.addConstr(quicksum((x_vars[s, n, g] for n in neighborhoods for g in groups)) <= school_capacities[s])
for s in schools:
    white_sum = quicksum((x_vars[s, n, 'White'] for n in neighborhoods))
    total_sum = quicksum((x_vars[s, n, g] for n in neighborhoods for g in groups))
    m.addConstr(white_sum >= min_white_ratio * total_sum)
    m.addConstr(white_sum <= max_white_ratio * total_sum)
m.setObjective(quicksum((distance[s, n] * x_vars[s, n, g] for s in schools for n in neighborhoods for g in groups)), GRB.MINIMIZE)
m.optimize()