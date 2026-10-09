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
for (idx, row) in school_capacity_df.iterrows():
    school = str(row['School'])
    try:
        cap = int(row['Capacity'])
    except Exception:
        raise ValueError(f"Invalid capacity for school {school}: {row['Capacity']}")
    school_capacity[school] = cap
neigh_population = {}
for (idx, row) in neigh_pop_df.iterrows():
    n = str(row['Neighborhood'])
    try:
        pop_white = int(row['Population_White'])
        pop_nonwhite = int(row['Population_NonWhite'])
    except Exception:
        raise ValueError(f"Invalid population for neighborhood {n}: {row['Population_White']}, {row['Population_NonWhite']}")
    neigh_population[n] = {'White': pop_white, 'NonWhite': pop_nonwhite}
distance = {}
distance_cols = [col for col in distance_df.columns if col != 'School']
for (idx, row) in distance_df.iterrows():
    school = str(row['School'])
    distance[school] = {}
    for n in neighborhoods:
        if n not in distance_cols:
            raise KeyError(f'Neighborhood {n} not found in distance.csv columns')
        try:
            dist = float(row[n])
        except Exception:
            raise ValueError(f'Invalid distance for school {school}, neighborhood {n}: {row[n]}')
        distance[school][n] = dist
total_white = sum((neigh_population[n]['White'] for n in neighborhoods))
total_nonwhite = sum((neigh_population[n]['NonWhite'] for n in neighborhoods))
total_students = total_white + total_nonwhite
district_white_ratio = 0.6
m = gp.Model('SchoolAssignment')
x_vars = m.addVars(schools, neighborhoods, groups, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((distance[s][n] * x_vars[s, n, g] for s in schools for n in neighborhoods for g in groups)), gp.GRB.MINIMIZE)
for n in neighborhoods:
    for g in groups:
        m.addConstr(gp.quicksum((x_vars[s, n, g] for s in schools)) == neigh_population[n][g], name=f'assign_{n}_{g}')
for s in schools:
    m.addConstr(gp.quicksum((x_vars[s, n, g] for n in neighborhoods for g in groups)) <= school_capacity[s], name=f'capacity_{s}')
for s in schools:
    white_sum = gp.quicksum((x_vars[s, n, 'White'] for n in neighborhoods))
    total_sum = gp.quicksum((x_vars[s, n, g] for n in neighborhoods for g in groups))
    m.addConstr(white_sum >= 0.5 * total_sum, name=f'racial_lb_{s}')
    m.addConstr(white_sum <= 0.7 * total_sum, name=f'racial_ub_{s}')
m.optimize()