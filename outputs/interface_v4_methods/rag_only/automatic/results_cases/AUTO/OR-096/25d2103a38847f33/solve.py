import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
school_capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/school_capacity.csv', sep=',')
neigh_pop_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/neighborhoods_population.csv', sep=',')
distance_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/distance.csv', sep=',')

def norm_id(x):
    return str(x).strip()
schools = [norm_id(s) for s in school_capacity_df['School'].unique()]
neighs = [norm_id(n) for n in neigh_pop_df['Neighborhood'].unique()]
groups = ['White', 'NonWhite']
school_capacity = {norm_id(row['School']): int(row['Capacity']) for _, row in school_capacity_df.iterrows()}
pop_white = {norm_id(row['Neighborhood']): int(row['Population_White']) for _, row in neigh_pop_df.iterrows()}
pop_nonwhite = {norm_id(row['Neighborhood']): int(row['Population_NonWhite']) for _, row in neigh_pop_df.iterrows()}
pop_by_group = {'White': pop_white, 'NonWhite': pop_nonwhite}
distance = {}
for _, row in distance_df.iterrows():
    s = norm_id(row['School'])
    for n in neighs:
        distance[s, n] = float(row[n])
for s in schools:
    if s not in school_capacity:
        raise ValueError(f'Missing capacity for school {s}')
for n in neighs:
    for g in groups:
        if n not in pop_by_group[g]:
            raise ValueError(f'Missing population for neighborhood {n}, group {g}')
for s in schools:
    for n in neighs:
        if (s, n) not in distance:
            raise ValueError(f'Missing distance for school {s}, neighborhood {n}')
district_white = sum(pop_white.values())
district_nonwhite = sum(pop_nonwhite.values())
district_total = district_white + district_nonwhite
district_white_ratio = district_white / district_total if district_total > 0 else 0.0
district_white_ratio = 0.6
district_nonwhite_ratio = 0.4
min_white_ratio = district_white_ratio - 0.1
max_white_ratio = district_white_ratio + 0.1
m = gp.Model('SchoolAssignment')
x = m.addVars(schools, neighs, groups, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((distance[s, n] * x[s, n, g] for s in schools for n in neighs for g in groups)), GRB.MINIMIZE)
for n in neighs:
    for g in groups:
        m.addConstr(gp.quicksum((x[s, n, g] for s in schools)) == pop_by_group[g][n], name='')
for s in schools:
    m.addConstr(gp.quicksum((x[s, n, g] for n in neighs for g in groups)) <= school_capacity[s], name='')
for s in schools:
    total_white = gp.quicksum((x[s, n, 'White'] for n in neighs))
    total_nonwhite = gp.quicksum((x[s, n, 'NonWhite'] for n in neighs))
    total_students = total_white + total_nonwhite
    m.addConstr(total_white >= min_white_ratio * total_students, name='')
    m.addConstr(total_white <= max_white_ratio * total_students, name='')
m.optimize()