import gurobipy as gp
import pandas as pd
import numpy as np
import re
school_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/school_capacity.csv'
neighborhoods_population_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/neighborhoods_population.csv'
distance_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/distance.csv'

def solve_school_assignment():
    school_capacity_df = pd.read_csv(school_capacity_path, sep=',', dtype=str, keep_default_na=False)
    neighborhoods_population_df = pd.read_csv(neighborhoods_population_path, sep=',', dtype=str, keep_default_na=False)
    distance_df = pd.read_csv(distance_path, sep=',', dtype=str, keep_default_na=False)
    schools = school_capacity_df['School'].tolist()
    if len(schools) != 2:
        raise ValueError('Query specifies exactly two schools; found: %s' % schools)
    school_capacity = {}
    for (_, row) in school_capacity_df.iterrows():
        school = row['School']
        try:
            cap = int(row['Capacity'])
        except Exception:
            raise ValueError(f"Invalid capacity for school {school}: {row['Capacity']}")
        school_capacity[school] = cap
    neighborhoods = neighborhoods_population_df['Neighborhood'].tolist()
    if len(neighborhoods) != 31:
        raise ValueError('Query specifies exactly 31 neighborhoods; found: %s' % neighborhoods)
    groups = ['White', 'NonWhite']
    population = {}
    for (_, row) in neighborhoods_population_df.iterrows():
        n = row['Neighborhood']
        try:
            w = int(row['Population_White'])
            nw = int(row['Population_NonWhite'])
        except Exception:
            raise ValueError(f"Invalid population for neighborhood {n}: {row['Population_White']}, {row['Population_NonWhite']}")
        population[n, 'White'] = w
        population[n, 'NonWhite'] = nw
    distance_cols = [col for col in distance_df.columns if col != 'School']
    if set(distance_cols) != set(neighborhoods):
        raise ValueError('Neighborhood columns in distance.csv do not match neighborhoods_population.csv')
    distance = {}
    for (_, row) in distance_df.iterrows():
        s = row['School']
        if s not in schools:
            raise ValueError(f'School {s} in distance.csv not found in school_capacity.csv')
        for n in neighborhoods:
            try:
                d = float(row[n])
            except Exception:
                raise ValueError(f'Invalid distance for school {s}, neighborhood {n}: {row[n]}')
            distance[s, n] = d
    for s in schools:
        for n in neighborhoods:
            if (s, n) not in distance:
                raise ValueError(f'Missing distance for school {s}, neighborhood {n}')
    for n in neighborhoods:
        for g in groups:
            if (n, g) not in population:
                raise ValueError(f'Missing population for neighborhood {n}, group {g}')
    m = gp.Model('SchoolAssignment')
    x_keys = [(s, n, g) for s in schools for n in neighborhoods for g in groups]
    x_vars = m.addVars(x_keys, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((distance[s, n] * x_vars[s, n, g] for s in schools for n in neighborhoods for g in groups)), gp.GRB.MINIMIZE)
    for n in neighborhoods:
        for g in groups:
            m.addConstr(gp.quicksum((x_vars[s, n, g] for s in schools)) == population[n, g], name=f'assign_{n}_{g}')
    for s in schools:
        m.addConstr(gp.quicksum((x_vars[s, n, g] for n in neighborhoods for g in groups)) <= school_capacity[s], name=f'capacity_{s}')
    for s in schools:
        W_s = gp.quicksum((x_vars[s, n, 'White'] for n in neighborhoods))
        T_s = gp.quicksum((x_vars[s, n, g] for n in neighborhoods for g in groups))
        m.addConstr(W_s >= 0.5 * T_s, name=f'racial_lb_{s}')
        m.addConstr(W_s <= 0.7 * T_s, name=f'racial_ub_{s}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_school_assignment()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')