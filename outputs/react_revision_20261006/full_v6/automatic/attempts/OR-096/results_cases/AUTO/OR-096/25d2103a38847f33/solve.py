import gurobipy as gp
import pandas as pd
import numpy as np
school_capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/school_capacity.csv', dtype=str, keep_default_na=False)
neigh_pop_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/neighborhoods_population.csv', dtype=str, keep_default_na=False)
distance_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/distance.csv', dtype=str, keep_default_na=False)
schools = [s.strip() for s in school_capacity_df['School'].unique()]
school_capacity = {}
for (_, row) in school_capacity_df.iterrows():
    school = row['School'].strip()
    try:
        cap = int(row['Capacity'])
    except Exception:
        raise ValueError(f"Invalid capacity for school {school}: {row['Capacity']}")
    school_capacity[school] = cap
neighborhoods = [n.strip() for n in neigh_pop_df['Neighborhood'].unique()]
neigh_population = {}
for (_, row) in neigh_pop_df.iterrows():
    n = row['Neighborhood'].strip()
    try:
        w = int(row['Population_White'])
        nw = int(row['Population_NonWhite'])
    except Exception:
        raise ValueError(f"Invalid population for neighborhood {n}: {row['Population_White']}, {row['Population_NonWhite']}")
    neigh_population[n] = {'White': w, 'NonWhite': nw}
groups = ['White', 'NonWhite']
distance = {}
distance_df = distance_df.set_index(distance_df.columns[0])
for school in schools:
    if school not in distance_df.index:
        raise ValueError(f'School {school} not found in distance.csv')
    row = distance_df.loc[school]
    distance[school] = {}
    for n in neighborhoods:
        if n not in row:
            raise ValueError(f'Neighborhood {n} not found in distance.csv columns')
        try:
            d = float(row[n])
        except Exception:
            raise ValueError(f'Invalid distance for school {school}, neighborhood {n}: {row[n]}')
        distance[school][n] = d
if set(schools) != set(distance.keys()):
    raise ValueError('Mismatch between schools in school_capacity.csv and distance.csv')
if set(neighborhoods) != set(neigh_population.keys()):
    raise ValueError('Mismatch between neighborhoods in neighborhoods_population.csv and distance.csv')
for s in schools:
    if set(distance[s].keys()) != set(neighborhoods):
        raise ValueError(f'Mismatch in neighborhoods for school {s} in distance.csv')
total_white = sum((neigh_population[n]['White'] for n in neighborhoods))
total_nonwhite = sum((neigh_population[n]['NonWhite'] for n in neighborhoods))
total_students = total_white + total_nonwhite
if total_students == 0:
    raise ValueError('Total number of students is zero.')
district_white_ratio = total_white / total_students
m = gp.Model('SchoolAssignment')
x_keys = []
for s in schools:
    for n in neighborhoods:
        for g in groups:
            x_keys.append((s, n, g))
x_vars = m.addVars(x_keys, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((distance[s][n] * x_vars[s, n, g] for s in schools for n in neighborhoods for g in groups)), gp.GRB.MINIMIZE)
for n in neighborhoods:
    for g in groups:
        pop = neigh_population[n][g]
        m.addConstr(gp.quicksum((x_vars[s, n, g] for s in schools)) == pop, name=f'assign_{n}_{g}')
for s in schools:
    m.addConstr(gp.quicksum((x_vars[s, n, g] for n in neighborhoods for g in groups)) <= school_capacity[s], name=f'capacity_{s}')
for s in schools:
    total_white_s = gp.quicksum((x_vars[s, n, 'White'] for n in neighborhoods))
    total_all_s = gp.quicksum((x_vars[s, n, g] for n in neighborhoods for g in groups))
    m.addConstr(total_white_s >= 0.5 * total_all_s, name=f'racial_lb_{s}')
    m.addConstr(total_white_s <= 0.7 * total_all_s, name=f'racial_ub_{s}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for key in x_keys:
        var = x_vars[key]
        print(f'{var.VarName} {var.X}')
else:
    print(f'Solver status: {m.Status}')