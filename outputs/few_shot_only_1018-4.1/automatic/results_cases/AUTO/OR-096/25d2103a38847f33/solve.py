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
school_id_col = 'School'
capacity_col = 'Capacity'
if 'School' not in school_capacity_df.columns:
    for col in school_capacity_df.columns:
        if re.fullmatch('\\s*School\\s*', col, re.IGNORECASE):
            school_id_col = col
        if re.fullmatch('\\s*Capacity\\s*', col, re.IGNORECASE):
            capacity_col = col
schools = [row[school_id_col].strip() for _, row in school_capacity_df.iterrows()]
school_capacity = {}
for _, row in school_capacity_df.iterrows():
    sid = row[school_id_col].strip()
    try:
        school_capacity[sid] = int(row[capacity_col])
    except Exception as e:
        raise ValueError(f'Invalid capacity for school {sid}: {row[capacity_col]}') from e
neigh_id_col = 'Neighborhood'
pop_white_col = 'Population_White'
pop_nonwhite_col = 'Population_NonWhite'
if 'Neighborhood' not in neighborhoods_population_df.columns:
    for col in neighborhoods_population_df.columns:
        if re.fullmatch('\\s*Neighborhood\\s*', col, re.IGNORECASE):
            neigh_id_col = col
        if re.fullmatch('\\s*Population_White\\s*', col, re.IGNORECASE):
            pop_white_col = col
        if re.fullmatch('\\s*Population_NonWhite\\s*', col, re.IGNORECASE):
            pop_nonwhite_col = col
neighborhoods = [row[neigh_id_col].strip() for _, row in neighborhoods_population_df.iterrows()]
groups = ['White', 'NonWhite']
pop_dict = {}
for _, row in neighborhoods_population_df.iterrows():
    n = row[neigh_id_col].strip()
    try:
        pop_dict[n, 'White'] = int(row[pop_white_col])
        pop_dict[n, 'NonWhite'] = int(row[pop_nonwhite_col])
    except Exception as e:
        raise ValueError(f'Invalid population for neighborhood {n}: {row[pop_white_col]}, {row[pop_nonwhite_col]}') from e
dist_school_col = 'School'
distance_neigh_cols = [col for col in distance_df.columns if col != dist_school_col]
distance_neigh_cols_norm = [col.strip() for col in distance_neigh_cols]
distance = {}
for _, row in distance_df.iterrows():
    s = row[dist_school_col].strip()
    for col, n in zip(distance_neigh_cols, distance_neigh_cols_norm):
        val = row[col].strip()
        try:
            distance[s, n] = float(val)
        except Exception as e:
            raise ValueError(f'Invalid distance for school {s}, neighborhood {n}: {val}') from e
for s in schools:
    for n in neighborhoods:
        if (s, n) not in distance:
            raise KeyError(f'Missing distance for school {s}, neighborhood {n}')
for n in neighborhoods:
    for g in groups:
        if (n, g) not in pop_dict:
            raise KeyError(f'Missing population for neighborhood {n}, group {g}')
total_white = sum((pop_dict[n, 'White'] for n in neighborhoods))
total_nonwhite = sum((pop_dict[n, 'NonWhite'] for n in neighborhoods))
total_students = total_white + total_nonwhite
district_white_ratio = total_white / total_students if total_students > 0 else 0.0
racial_lb = 0.5
racial_ub = 0.7
m = gp.Model('SchoolAssignment')
x_vars = m.addVars(schools, neighborhoods, groups, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((distance[s, n] * x_vars[s, n, g] for s in schools for n in neighborhoods for g in groups)), gp.GRB.MINIMIZE)
for n in neighborhoods:
    for g in groups:
        m.addConstr(gp.quicksum((x_vars[s, n, g] for s in schools)) == pop_dict[n, g], name=f'assign_{n}_{g}')
for s in schools:
    m.addConstr(gp.quicksum((x_vars[s, n, g] for n in neighborhoods for g in groups)) <= school_capacity[s], name=f'capacity_{s}')
for s in schools:
    total_assigned = gp.quicksum((x_vars[s, n, g] for n in neighborhoods for g in groups))
    total_white_assigned = gp.quicksum((x_vars[s, n, 'White'] for n in neighborhoods))
    m.addConstr(total_white_assigned >= racial_lb * total_assigned, name=f'racial_lb_{s}')
    m.addConstr(total_white_assigned <= racial_ub * total_assigned, name=f'racial_ub_{s}')
m.optimize()