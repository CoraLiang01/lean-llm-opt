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
if set(schools) != set(distance_df.index):
    raise ValueError('Mismatch between schools in school_capacity.csv and distance.csv')
if set(neighborhoods) - set(distance_df.columns):
    raise ValueError('Some neighborhoods in neighborhoods_population.csv are missing in distance.csv')
distance = {}
for s in schools:
    for n in neighborhoods:
        distance[s, n] = float(distance_df.loc[s, n])
groups = ['White', 'NonWhite']
m = Model('SchoolAssignment')
x = m.addVars(schools, neighborhoods, groups, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(quicksum((distance[s, n] * x[s, n, g] for s in schools for n in neighborhoods for g in groups)), GRB.MINIMIZE)
for n in neighborhoods:
    m.addConstr(quicksum((x[s, n, 'White'] for s in schools)) == int(pop_white[n]), name=f'assign_white_{n}')
    m.addConstr(quicksum((x[s, n, 'NonWhite'] for s in schools)) == int(pop_nonwhite[n]), name=f'assign_nonwhite_{n}')
for s in schools:
    m.addConstr(quicksum((x[s, n, g] for n in neighborhoods for g in groups)) <= int(school_capacities[s]), name=f'capacity_{s}')
for s in schools:
    W_s = quicksum((x[s, n, 'White'] for n in neighborhoods))
    T_s = quicksum((x[s, n, g] for n in neighborhoods for g in groups))
    m.addConstr(W_s >= 0.5 * T_s, name=f'racial_lb_{s}')
    m.addConstr(W_s <= 0.7 * T_s, name=f'racial_ub_{s}')
m.optimize()