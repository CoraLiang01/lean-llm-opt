import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
school_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/school_capacity.csv'
neighborhoods_population_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/neighborhoods_population.csv'
distance_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/distance.csv'
school_capacity_df = pd.read_csv(school_capacity_path, sep=',')
neighborhoods_population_df = pd.read_csv(neighborhoods_population_path, sep=',')
distance_df = pd.read_csv(distance_path, sep=',')
school_capacity_df['School'] = school_capacity_df['School'].astype(str).str.strip()
distance_df['School'] = distance_df['School'].astype(str).str.strip()
neighborhoods_population_df['Neighborhood'] = neighborhoods_population_df['Neighborhood'].astype(str).str.strip()
distance_df.columns = [col.strip() if col != 'School' else col for col in distance_df.columns]
schools = list(school_capacity_df['School'].unique())
neighborhoods = list(neighborhoods_population_df['Neighborhood'].unique())
groups = ['White', 'NonWhite']
school_capacity = {}
for (_, row) in school_capacity_df.iterrows():
    school = str(row['School']).strip()
    school_capacity[school] = int(row['Capacity'])
pop_white = {}
pop_nonwhite = {}
for (_, row) in neighborhoods_population_df.iterrows():
    n = str(row['Neighborhood']).strip()
    pop_white[n] = int(row['Population_White'])
    pop_nonwhite[n] = int(row['Population_NonWhite'])
pop = {}
for n in neighborhoods:
    pop[n, 'White'] = pop_white[n]
    pop[n, 'NonWhite'] = pop_nonwhite[n]
distance = {}
for (_, row) in distance_df.iterrows():
    s = str(row['School']).strip()
    for n in neighborhoods:
        if n not in distance_df.columns:
            raise ValueError(f'Neighborhood {n} not found in distance.csv columns')
        val = row[n]
        if pd.isnull(val):
            raise ValueError(f'Missing distance for school {s}, neighborhood {n}')
        distance[s, n] = float(val)
for s in schools:
    if s not in school_capacity:
        raise ValueError(f'Missing capacity for school {s}')
for n in neighborhoods:
    for g in groups:
        if (n, g) not in pop:
            raise ValueError(f'Missing population for neighborhood {n}, group {g}')
for s in schools:
    for n in neighborhoods:
        if (s, n) not in distance:
            raise ValueError(f'Missing distance for school {s}, neighborhood {n}')
district_white = sum((pop[n, 'White'] for n in neighborhoods))
district_nonwhite = sum((pop[n, 'NonWhite'] for n in neighborhoods))
district_total = district_white + district_nonwhite
district_white_ratio = district_white / district_total if district_total > 0 else 0.0
target_white_ratio = 0.6
lower_white_ratio = 0.5
upper_white_ratio = 0.7
m = gp.Model('SchoolAssignment')
m.setParam('MIPGap', 0.0001)
x_keys = []
for s in schools:
    for n in neighborhoods:
        for g in groups:
            x_keys.append((s, n, g))
x = m.addVars(x_keys, vtype=GRB.INTEGER, lb=0, name='')
obj = gp.LinExpr()
for s in schools:
    for n in neighborhoods:
        for g in groups:
            obj.addTerms(distance[s, n], x[s, n, g])
m.setObjective(obj, GRB.MINIMIZE)
for n in neighborhoods:
    for g in groups:
        m.addConstr(gp.quicksum((x[s, n, g] for s in schools)) == pop[n, g], name='')
for s in schools:
    m.addConstr(gp.quicksum((x[s, n, g] for n in neighborhoods for g in groups)) <= school_capacity[s], name='')
for s in schools:
    total_white = gp.quicksum((x[s, n, 'White'] for n in neighborhoods))
    total_students = gp.quicksum((x[s, n, g] for n in neighborhoods for g in groups))
    m.addConstr(total_white >= lower_white_ratio * total_students, name='')
    m.addConstr(total_white <= upper_white_ratio * total_students, name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.Status}')