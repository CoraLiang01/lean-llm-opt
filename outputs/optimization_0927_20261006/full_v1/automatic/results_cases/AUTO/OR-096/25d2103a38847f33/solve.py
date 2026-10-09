import gurobipy as gp
import pandas as pd
import numpy as np
import re
school_capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/school_capacity.csv', dtype=str, keep_default_na=False)
neigh_pop_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/neighborhoods_population.csv', dtype=str, keep_default_na=False)
distance_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/distance.csv', dtype=str, keep_default_na=False)
schools = [s.strip() for s in school_capacity_df['School']]
neighborhoods = [n.strip() for n in neigh_pop_df['Neighborhood']]
groups = ['White', 'NonWhite']
school_capacity = {}
for (idx, row) in school_capacity_df.iterrows():
    school = row['School'].strip()
    try:
        cap = int(row['Capacity'])
    except Exception:
        raise ValueError(f"Invalid capacity for school {school}: {row['Capacity']}")
    school_capacity[school] = cap
neigh_population = {}
for (idx, row) in neigh_pop_df.iterrows():
    neigh = row['Neighborhood'].strip()
    try:
        pop_white = int(row['Population_White'])
        pop_nonwhite = int(row['Population_NonWhite'])
    except Exception:
        raise ValueError(f"Invalid population for neighborhood {neigh}: {row['Population_White']}, {row['Population_NonWhite']}")
    neigh_population[neigh] = {'White': pop_white, 'NonWhite': pop_nonwhite}
distance = {}
for (idx, row) in distance_df.iterrows():
    school = row['School'].strip()
    distance[school] = {}
    for neigh in neighborhoods:
        val = row[neigh]
        try:
            dist = float(val)
        except Exception:
            raise ValueError(f'Invalid distance for school {school}, neighborhood {neigh}: {val}')
        distance[school][neigh] = dist
district_white = sum((neigh_population[n]['White'] for n in neighborhoods))
district_nonwhite = sum((neigh_population[n]['NonWhite'] for n in neighborhoods))
district_total = district_white + district_nonwhite
if district_total == 0:
    raise ValueError('Total district population is zero.')
district_white_ratio = district_white / district_total
m = gp.Model('SchoolAssignment')
x_vars = m.addVars(schools, neighborhoods, groups, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((distance[s][n] * x_vars[s, n, g] for s in schools for n in neighborhoods for g in groups)), gp.GRB.MINIMIZE)
for n in neighborhoods:
    for g in groups:
        m.addConstr(gp.quicksum((x_vars[s, n, g] for s in schools)) == neigh_population[n][g], name=f'assign_{n}_{g}')
for s in schools:
    m.addConstr(gp.quicksum((x_vars[s, n, g] for n in neighborhoods for g in groups)) <= school_capacity[s], name=f'capacity_{s}')
for s in schools:
    total_white = gp.quicksum((x_vars[s, n, 'White'] for n in neighborhoods))
    total_all = gp.quicksum((x_vars[s, n, g] for n in neighborhoods for g in groups))
    m.addConstr(total_white >= 0.5 * total_all, name=f'racial_lb_{s}')
    m.addConstr(total_white <= 0.7 * total_all, name=f'racial_ub_{s}')
m.optimize()