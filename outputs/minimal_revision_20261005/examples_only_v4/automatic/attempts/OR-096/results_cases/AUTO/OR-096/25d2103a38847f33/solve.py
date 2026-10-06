import gurobipy as gp
import pandas as pd
import numpy as np
school_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/school_capacity.csv'
neigh_pop_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/neighborhoods_population.csv'
distance_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/distance.csv'
school_df = pd.read_csv(school_capacity_path, sep=',')
schools = list(school_df['School'].astype(str))
school_cap = dict(zip(school_df['School'].astype(str), school_df['Capacity']))
neigh_df = pd.read_csv(neigh_pop_path, sep=',')
neighs = list(neigh_df['Neighborhood'].astype(str))
pop_white = dict(zip(neigh_df['Neighborhood'].astype(str), neigh_df['Population_White']))
pop_nonwhite = dict(zip(neigh_df['Neighborhood'].astype(str), neigh_df['Population_NonWhite']))
groups = ['White', 'NonWhite']
total_white = sum((pop_white[n] for n in neighs))
total_nonwhite = sum((pop_nonwhite[n] for n in neighs))
total_students = total_white + total_nonwhite
white_lb = 0.5
white_ub = 0.7
dist_df = pd.read_csv(distance_path, sep=',')
dist_df['School'] = dist_df['School'].astype(str)
dist_dict = {}
for (_, row) in dist_df.iterrows():
    s = row['School']
    for n in neighs:
        dist_dict[s, n] = float(row[n])
for s in schools:
    for n in neighs:
        if (s, n) not in dist_dict:
            raise ValueError(f'Missing distance for school {s}, neighborhood {n}')
for n in neighs:
    if n not in pop_white or n not in pop_nonwhite:
        raise ValueError(f'Missing population data for neighborhood {n}')
var_keys = []
for s in schools:
    for n in neighs:
        for g in groups:
            var_keys.append((s, n, g))
m = gp.Model('SchoolAssignment')
m.Params.MIPGap = 0.0001
x = m.addVars(var_keys, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((dist_dict[s, n] * x[s, n, g] for (s, n, g) in var_keys)), gp.GRB.MINIMIZE)
for n in neighs:
    m.addConstr(gp.quicksum((x[s, n, 'White'] for s in schools)) == pop_white[n], name=f'assign_white_{n}')
    m.addConstr(gp.quicksum((x[s, n, 'NonWhite'] for s in schools)) == pop_nonwhite[n], name=f'assign_nonwhite_{n}')
for s in schools:
    m.addConstr(gp.quicksum((x[s, n, g] for n in neighs for g in groups)) <= school_cap[s], name=f'cap_{s}')
for s in schools:
    total_white_s = gp.quicksum((x[s, n, 'White'] for n in neighs))
    total_students_s = gp.quicksum((x[s, n, g] for n in neighs for g in groups))
    m.addConstr(total_white_s >= white_lb * total_students_s, name=f'race_lb_{s}')
    m.addConstr(total_white_s <= white_ub * total_students_s, name=f'race_ub_{s}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for (s, n, g) in var_keys:
        v = x[s, n, g]
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')