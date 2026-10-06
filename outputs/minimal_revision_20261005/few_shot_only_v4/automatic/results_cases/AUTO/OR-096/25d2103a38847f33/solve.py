import gurobipy as gp
import pandas as pd
import numpy as np
import re
school_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/school_capacity.csv'
neighborhoods_population_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/neighborhoods_population.csv'
distance_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/distance.csv'

def solve_school_assignment():
    school_cap_df = pd.read_csv(school_capacity_path, sep=',')
    neigh_pop_df = pd.read_csv(neighborhoods_population_path, sep=',')
    dist_df = pd.read_csv(distance_path, sep=',')
    schools = [str(s) for s in school_cap_df['School'].unique()]
    neighborhoods = [str(n) for n in neigh_pop_df['Neighborhood'].unique()]
    groups = ['White', 'NonWhite']
    school_capacity = {}
    for (_, row) in school_cap_df.iterrows():
        school = str(row['School'])
        cap = float(row['Capacity'])
        if school in school_capacity:
            raise ValueError(f"Duplicate school '{school}' in school_capacity.csv")
        school_capacity[school] = cap
    if set(schools) != set(school_capacity.keys()):
        raise ValueError('Mismatch in school identifiers between index set and school_capacity.csv')
    pop_white = {}
    pop_nonwhite = {}
    for (_, row) in neigh_pop_df.iterrows():
        n = str(row['Neighborhood'])
        pop_white[n] = float(row['Population_White'])
        pop_nonwhite[n] = float(row['Population_NonWhite'])
    if set(neighborhoods) != set(pop_white.keys()) or set(neighborhoods) != set(pop_nonwhite.keys()):
        raise ValueError('Mismatch in neighborhood identifiers between index set and neighborhoods_population.csv')
    dist = {}
    dist_df['School'] = dist_df['School'].astype(str)
    dist_cols = [c for c in dist_df.columns if c != 'School']
    for (_, row) in dist_df.iterrows():
        school = str(row['School'])
        if school not in schools:
            continue
        for n in dist_cols:
            neigh = str(n)
            if neigh not in neighborhoods:
                continue
            val = row[n]
            if pd.isnull(val):
                raise ValueError(f'Missing distance for school {school}, neighborhood {neigh}')
            dist[school, neigh] = float(val)
    for s in schools:
        for n in neighborhoods:
            if (s, n) not in dist:
                raise ValueError(f'Missing distance for school {s}, neighborhood {n} in distance.csv')
    total_white = sum((pop_white[n] for n in neighborhoods))
    total_nonwhite = sum((pop_nonwhite[n] for n in neighborhoods))
    total_students = total_white + total_nonwhite
    if total_students <= 0:
        raise ValueError('Total number of students is zero.')
    keys = []
    for s in schools:
        for n in neighborhoods:
            for g in groups:
                keys.append((s, n, g))
    m = gp.Model('SchoolAssignment')
    x = m.addVars(keys, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    obj = gp.LinExpr()
    for s in schools:
        for n in neighborhoods:
            for g in groups:
                obj += dist[s, n] * x[s, n, g]
    m.setObjective(obj, gp.GRB.MINIMIZE)
    for n in neighborhoods:
        m.addConstr(gp.quicksum((x[s, n, 'White'] for s in schools)) == pop_white[n], name=f'assign_white_{n}')
        m.addConstr(gp.quicksum((x[s, n, 'NonWhite'] for s in schools)) == pop_nonwhite[n], name=f'assign_nonwhite_{n}')
    for s in schools:
        m.addConstr(gp.quicksum((x[s, n, g] for n in neighborhoods for g in groups)) <= school_capacity[s], name=f'capacity_{s}')
    for s in schools:
        total_assigned = gp.quicksum((x[s, n, g] for n in neighborhoods for g in groups))
        total_white_assigned = gp.quicksum((x[s, n, 'White'] for n in neighborhoods))
        m.addConstr(total_white_assigned >= 0.5 * total_assigned, name=f'racial_lb_{s}')
        m.addConstr(total_white_assigned <= 0.7 * total_assigned, name=f'racial_ub_{s}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_school_assignment()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')