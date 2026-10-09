import gurobipy as gp
import pandas as pd
import numpy as np
import re

def solve_school_assignment():
    school_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/school_capacity.csv'
    neighborhoods_population_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/neighborhoods_population.csv'
    distance_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/distance.csv'
    school_df = pd.read_csv(school_capacity_path, sep=',', dtype=str, keep_default_na=False)
    neighborhoods_df = pd.read_csv(neighborhoods_population_path, sep=',', dtype=str, keep_default_na=False)
    distance_df = pd.read_csv(distance_path, sep=',', dtype=str, keep_default_na=False)
    schools = school_df['School'].tolist()
    neighborhoods = neighborhoods_df['Neighborhood'].tolist()
    groups = ['White', 'NonWhite']
    school_capacity = {}
    for (idx, row) in school_df.iterrows():
        school = str(row['School'])
        try:
            cap = int(row['Capacity'])
        except Exception:
            raise ValueError(f"Invalid Capacity for school {school}: {row['Capacity']}")
        school_capacity[school] = cap
    pop_white = {}
    pop_nonwhite = {}
    for (idx, row) in neighborhoods_df.iterrows():
        n = str(row['Neighborhood'])
        try:
            pop_white[n] = int(row['Population_White'])
            pop_nonwhite[n] = int(row['Population_NonWhite'])
        except Exception:
            raise ValueError(f"Invalid population for neighborhood {n}: {row['Population_White']}, {row['Population_NonWhite']}")
    distance_neigh_cols = [col for col in distance_df.columns if col != 'School']
    if set(neighborhoods) != set(distance_neigh_cols):
        raise ValueError(f'Neighborhoods in population and distance files do not match: {set(neighborhoods) ^ set(distance_neigh_cols)}')
    distance = {}
    for (idx, row) in distance_df.iterrows():
        school = str(row['School'])
        if school not in schools:
            continue
        for n in neighborhoods:
            try:
                d = float(row[n])
            except Exception:
                raise ValueError(f'Invalid distance for school {school}, neighborhood {n}: {row[n]}')
            distance[school, n] = d
    total_white = sum((pop_white[n] for n in neighborhoods))
    total_nonwhite = sum((pop_nonwhite[n] for n in neighborhoods))
    total_students = total_white + total_nonwhite
    district_white_ratio = total_white / total_students if total_students > 0 else 0.0
    x_keys = []
    for s in schools:
        for n in neighborhoods:
            for g in groups:
                x_keys.append((s, n, g))
    m = gp.Model('SchoolAssignment')
    m.setParam('MIPGap', 0.0001)
    x_vars = m.addVars(x_keys, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((x_vars[s, n, g] * distance[s, n] for (s, n, g) in x_keys)), gp.GRB.MINIMIZE)
    for n in neighborhoods:
        m.addConstr(gp.quicksum((x_vars[s, n, 'White'] for s in schools)) == pop_white[n], name=f'assign_white_{n}')
        m.addConstr(gp.quicksum((x_vars[s, n, 'NonWhite'] for s in schools)) == pop_nonwhite[n], name=f'assign_nonwhite_{n}')
    for s in schools:
        m.addConstr(gp.quicksum((x_vars[s, n, g] for n in neighborhoods for g in groups)) <= school_capacity[s], name=f'capacity_{s}')
    for s in schools:
        total_white_s = gp.quicksum((x_vars[s, n, 'White'] for n in neighborhoods))
        total_all_s = gp.quicksum((x_vars[s, n, g] for n in neighborhoods for g in groups))
        m.addConstr(total_white_s >= 0.5 * total_all_s, name=f'racial_lb_{s}')
        m.addConstr(total_white_s <= 0.7 * total_all_s, name=f'racial_ub_{s}')
    m.optimize()
    return m
m = solve_school_assignment()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')