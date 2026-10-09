import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
school_capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/school_capacity.csv', sep=',')
school_capacity_df['School'] = school_capacity_df['School'].astype(str).str.strip()
schools = list(school_capacity_df['School'])
school_capacities = dict(zip(school_capacity_df['School'], school_capacity_df['Capacity']))
neigh_pop_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/neighborhoods_population.csv', sep=',')
neigh_pop_df['Neighborhood'] = neigh_pop_df['Neighborhood'].astype(str).str.strip()
neighborhoods = list(neigh_pop_df['Neighborhood'])
pop_white = dict(zip(neigh_pop_df['Neighborhood'], neigh_pop_df['Population_White']))
pop_nonwhite = dict(zip(neigh_pop_df['Neighborhood'], neigh_pop_df['Population_NonWhite']))
distance_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/distance.csv', sep=',')
distance_df['School'] = distance_df['School'].astype(str).str.strip()
distance_df.set_index('School', inplace=True)
if not set(schools).issubset(distance_df.index):
    raise ValueError('Some schools in school_capacity.csv are missing from distance.csv')
if not set(neighborhoods).issubset(distance_df.columns):
    raise ValueError('Some neighborhoods in neighborhoods_population.csv are missing from distance.csv')
distance = {}
for s in schools:
    for n in neighborhoods:
        val = distance_df.loc[s, n]
        if pd.isnull(val):
            raise ValueError(f'Missing distance for school {s}, neighborhood {n}')
        distance[s, n] = float(val)
groups = ['White', 'NonWhite']
m = Model('SchoolAssignment')
x = m.addVars(schools, neighborhoods, groups, lb=0.0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(quicksum((distance[s, n] * x[s, n, g] for s in schools for n in neighborhoods for g in groups)), GRB.MINIMIZE)
for n in neighborhoods:
    m.addConstr(quicksum((x[s, n, 'White'] for s in schools)) == pop_white[n])
    m.addConstr(quicksum((x[s, n, 'NonWhite'] for s in schools)) == pop_nonwhite[n])
for s in schools:
    m.addConstr(quicksum((x[s, n, g] for n in neighborhoods for g in groups)) <= school_capacities[s])
for s in schools:
    total_white = quicksum((x[s, n, 'White'] for n in neighborhoods))
    total_nonwhite = quicksum((x[s, n, 'NonWhite'] for n in neighborhoods))
    total_students = total_white + total_nonwhite
    m.addConstr(total_white >= 0.5 * (total_white + total_nonwhite))
    m.addConstr(total_white <= 0.7 * (total_white + total_nonwhite))
m.optimize()