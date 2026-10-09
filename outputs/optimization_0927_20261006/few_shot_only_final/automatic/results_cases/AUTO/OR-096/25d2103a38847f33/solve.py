import gurobipy as gp
import pandas as pd
import numpy as np
import re
school_capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/school_capacity.csv', dtype=str, keep_default_na=False)
neigh_pop_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/neighborhoods_population.csv', dtype=str, keep_default_na=False)
distance_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/distance.csv', dtype=str, keep_default_na=False)
schools = school_capacity_df['School'].tolist()
neighborhoods = neigh_pop_df['Neighborhood'].tolist()
groups = ['White', 'NonWhite']
school_capacity = {}
for (_, row) in school_capacity_df.iterrows():
    school = row['School']
    try:
        cap = int(row['Capacity'])
    except Exception:
        raise ValueError(f"Invalid capacity for school {school}: {row['Capacity']}")
    school_capacity[school] = cap
neigh_population = {}
for (_, row) in neigh_pop_df.iterrows():
    n = row['Neighborhood']
    try:
        w = int(row['Population_White'])
        nw = int(row['Population_NonWhite'])
    except Exception:
        raise ValueError(f"Invalid population for neighborhood {n}: {row['Population_White']}, {row['Population_NonWhite']}")
    neigh_population[n, 'White'] = w
    neigh_population[n, 'NonWhite'] = nw
total_white = sum((neigh_population[n, 'White'] for n in neighborhoods))
total_nonwhite = sum((neigh_population[n, 'NonWhite'] for n in neighborhoods))
total_students = total_white + total_nonwhite
min_white_ratio = 0.5
max_white_ratio = 0.7
distance = {}
for (_, row) in distance_df.iterrows():
    school = row['School']
    for n in neighborhoods:
        if n not in row:
            raise KeyError(f'Neighborhood {n} not found in distance.csv columns')
        try:
            d = float(row[n])
        except Exception:
            raise ValueError(f'Invalid distance for school {school}, neighborhood {n}: {row[n]}')
        distance[school, n] = d
m = gp.Model('SchoolAssignment')
x_vars = m.addVars(schools, neighborhoods, groups, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((distance[s, n] * x_vars[s, n, g] for s in schools for n in neighborhoods for g in groups)), gp.GRB.MINIMIZE)
for n in neighborhoods:
    for g in groups:
        m.addConstr(gp.quicksum((x_vars[s, n, g] for s in schools)) == neigh_population[n, g], name=f'assign_{n}_{g}')
for s in schools:
    m.addConstr(gp.quicksum((x_vars[s, n, g] for n in neighborhoods for g in groups)) <= school_capacity[s], name=f'capacity_{s}')
for s in schools:
    total_white_expr = gp.quicksum((x_vars[s, n, 'White'] for n in neighborhoods))
    total_students_expr = gp.quicksum((x_vars[s, n, g] for n in neighborhoods for g in groups))
    m.addConstr(total_white_expr >= min_white_ratio * total_students_expr, name=f'min_white_ratio_{s}')
    m.addConstr(total_white_expr <= max_white_ratio * total_students_expr, name=f'max_white_ratio_{s}')
m.optimize()