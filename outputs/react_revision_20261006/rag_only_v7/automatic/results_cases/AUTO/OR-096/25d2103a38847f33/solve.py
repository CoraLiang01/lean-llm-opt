import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
school_capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/school_capacity.csv', dtype=str, keep_default_na=False)
neighborhoods_population_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/neighborhoods_population.csv', dtype=str, keep_default_na=False)
distance_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/distance.csv', dtype=str, keep_default_na=False)
school_capacity_df['School'] = school_capacity_df['School'].str.strip()
school_ids = list(school_capacity_df['School'])
neighborhoods_population_df['Neighborhood'] = neighborhoods_population_df['Neighborhood'].str.strip()
neighborhood_ids = list(neighborhoods_population_df['Neighborhood'])
groups = ['White', 'NonWhite']
school_capacity = {}
for (_, row) in school_capacity_df.iterrows():
    school = row['School']
    try:
        cap = int(row['Capacity'])
    except Exception:
        raise ValueError(f"Invalid capacity for school {school}: {row['Capacity']}")
    school_capacity[school] = cap
population_white = {}
population_nonwhite = {}
for (_, row) in neighborhoods_population_df.iterrows():
    n = row['Neighborhood']
    try:
        w = int(row['Population_White'])
        nw = int(row['Population_NonWhite'])
    except Exception:
        raise ValueError(f"Invalid population for neighborhood {n}: {row['Population_White']}, {row['Population_NonWhite']}")
    population_white[n] = w
    population_nonwhite[n] = nw
distance = {}
distance_df['School'] = distance_df['School'].str.strip()
for (_, row) in distance_df.iterrows():
    school = row['School']
    for n in neighborhood_ids:
        if n not in row:
            raise ValueError(f'Neighborhood {n} missing in distance.csv for school {school}')
        try:
            d = float(row[n])
        except Exception:
            raise ValueError(f'Invalid distance for school {school}, neighborhood {n}: {row[n]}')
        distance[school, n] = d
for s in school_ids:
    if s not in school_capacity:
        raise ValueError(f'School {s} missing in school_capacity.csv')
for n in neighborhood_ids:
    if n not in population_white or n not in population_nonwhite:
        raise ValueError(f'Neighborhood {n} missing in neighborhoods_population.csv')
    for s in school_ids:
        if (s, n) not in distance:
            raise ValueError(f'Distance missing for school {s}, neighborhood {n}')
x_keys = []
for s in school_ids:
    for n in neighborhood_ids:
        for g in groups:
            x_keys.append((s, n, g))
m = gp.Model('school_assignment')
x_vars = m.addVars(x_keys, vtype=GRB.INTEGER, lb=0, name='')
obj = gp.quicksum((distance[s, n] * x_vars[s, n, g] for s in school_ids for n in neighborhood_ids for g in groups))
m.setObjective(obj, GRB.MINIMIZE)
for n in neighborhood_ids:
    m.addConstr(gp.quicksum((x_vars[s, n, 'White'] for s in school_ids)) == population_white[n], name=f'assign_white_{n}')
    m.addConstr(gp.quicksum((x_vars[s, n, 'NonWhite'] for s in school_ids)) == population_nonwhite[n], name=f'assign_nonwhite_{n}')
for s in school_ids:
    m.addConstr(gp.quicksum((x_vars[s, n, g] for n in neighborhood_ids for g in groups)) <= school_capacity[s], name=f'capacity_{s}')
for s in school_ids:
    total_white = gp.quicksum((x_vars[s, n, 'White'] for n in neighborhood_ids))
    total_students = gp.quicksum((x_vars[s, n, g] for n in neighborhood_ids for g in groups))
    m.addConstr(total_white >= 0.5 * total_students, name=f'racial_lb_{s}')
    m.addConstr(total_white <= 0.7 * total_students, name=f'racial_ub_{s}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for key in x_keys:
        var = x_vars[key]
        print(f'{var.VarName} {var.X}')
else:
    print(f'Solver status: {m.Status}')