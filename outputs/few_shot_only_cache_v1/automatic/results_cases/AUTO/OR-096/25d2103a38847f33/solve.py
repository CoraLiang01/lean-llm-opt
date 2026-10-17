import gurobipy as gp
import pandas as pd
import numpy as np
import re
school_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/school_capacity.csv'
neighborhoods_population_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/neighborhoods_population.csv'
distance_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/distance.csv'
school_capacity_df = pd.read_csv(school_capacity_path, dtype=str, keep_default_na=False)
neighborhoods_population_df = pd.read_csv(neighborhoods_population_path, dtype=str, keep_default_na=False)
distance_df = pd.read_csv(distance_path, dtype=str, keep_default_na=False)
schools = school_capacity_df['School'].tolist()
neighborhoods = neighborhoods_population_df['Neighborhood'].tolist()
groups = ['White', 'NonWhite']
school_capacity = {}
for _, row in school_capacity_df.iterrows():
    school = row['School']
    try:
        cap = int(row['Capacity'])
    except Exception as e:
        raise ValueError(f"Invalid capacity for school {school}: {row['Capacity']}")
    school_capacity[school] = cap
neighborhood_population = {n: {} for n in neighborhoods}
for _, row in neighborhoods_population_df.iterrows():
    n = row['Neighborhood']
    try:
        w = int(row['Population_White'])
        nw = int(row['Population_NonWhite'])
    except Exception as e:
        raise ValueError(f"Invalid population for neighborhood {n}: {row['Population_White']}, {row['Population_NonWhite']}")
    neighborhood_population[n]['White'] = w
    neighborhood_population[n]['NonWhite'] = nw
distance = {s: {} for s in schools}
for _, row in distance_df.iterrows():
    school = row['School']
    if school not in schools:
        continue
    for n in neighborhoods:
        try:
            d = float(row[n])
        except Exception as e:
            raise ValueError(f'Invalid distance for school {school}, neighborhood {n}: {row[n]}')
        distance[school][n] = d
total_white = sum((neighborhood_population[n]['White'] for n in neighborhoods))
total_nonwhite = sum((neighborhood_population[n]['NonWhite'] for n in neighborhoods))
total_students = total_white + total_nonwhite
m = gp.Model('SchoolAssignment')
x_vars = m.addVars(schools, neighborhoods, groups, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((x_vars[s, n, g] * distance[s][n] for s in schools for n in neighborhoods for g in groups)), gp.GRB.MINIMIZE)
for n in neighborhoods:
    for g in groups:
        m.addConstr(gp.quicksum((x_vars[s, n, g] for s in schools)) == neighborhood_population[n][g], name=f'assign_{n}_{g}')
for s in schools:
    m.addConstr(gp.quicksum((x_vars[s, n, g] for n in neighborhoods for g in groups)) <= school_capacity[s], name=f'capacity_{s}')
for s in schools:
    total_white_s = gp.quicksum((x_vars[s, n, 'White'] for n in neighborhoods))
    total_all_s = gp.quicksum((x_vars[s, n, g] for n in neighborhoods for g in groups))
    m.addConstr(total_white_s >= 0.5 * total_all_s, name=f'racial_lb_{s}')
    m.addConstr(total_white_s <= 0.7 * total_all_s, name=f'racial_ub_{s}')
m.optimize()