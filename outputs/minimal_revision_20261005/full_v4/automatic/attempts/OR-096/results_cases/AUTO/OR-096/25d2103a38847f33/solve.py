import gurobipy as gp
import pandas as pd
import numpy as np
school_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/school_capacity.csv'
neighborhoods_population_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/neighborhoods_population.csv'
distance_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/distance.csv'

def solve_problem():
    school_df = pd.read_csv(school_capacity_path, sep=',')
    neighborhoods_df = pd.read_csv(neighborhoods_population_path, sep=',')
    distance_df = pd.read_csv(distance_path, sep=',')
    schools = [str(s).strip() for s in school_df['School']]
    neighborhoods = [str(n).strip() for n in neighborhoods_df['Neighborhood']]
    groups = ['White', 'NonWhite']
    school_capacity = {}
    for (_, row) in school_df.iterrows():
        school = str(row['School']).strip()
        school_capacity[school] = int(row['Capacity'])
    pop_white = {}
    pop_nonwhite = {}
    for (_, row) in neighborhoods_df.iterrows():
        n = str(row['Neighborhood']).strip()
        pop_white[n] = int(row['Population_White'])
        pop_nonwhite[n] = int(row['Population_NonWhite'])
    distance_neigh_cols = [c for c in distance_df.columns if c != 'School']
    if set(neighborhoods) != set(distance_neigh_cols):
        raise ValueError('Neighborhoods in population and distance files do not match.')
    distance = {}
    for (_, row) in distance_df.iterrows():
        school = str(row['School']).strip()
        for n in neighborhoods:
            distance[school, n] = float(row[n])
    total_white = sum((pop_white[n] for n in neighborhoods))
    total_nonwhite = sum((pop_nonwhite[n] for n in neighborhoods))
    total_students = total_white + total_nonwhite
    district_white_ratio = total_white / total_students if total_students > 0 else 0.0
    keys = []
    for s in schools:
        for n in neighborhoods:
            for g in groups:
                keys.append((s, n, g))
    m = gp.Model('SchoolAssignment')
    x = m.addVars(keys, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    obj = gp.quicksum((distance[s, n] * x[s, n, g] for s in schools for n in neighborhoods for g in groups))
    m.setObjective(obj, gp.GRB.MINIMIZE)
    for n in neighborhoods:
        m.addConstr(gp.quicksum((x[s, n, 'White'] for s in schools)) == pop_white[n], name=f'assign_white_{n}')
        m.addConstr(gp.quicksum((x[s, n, 'NonWhite'] for s in schools)) == pop_nonwhite[n], name=f'assign_nonwhite_{n}')
    for s in schools:
        m.addConstr(gp.quicksum((x[s, n, g] for n in neighborhoods for g in groups)) <= school_capacity[s], name=f'capacity_{s}')
    for s in schools:
        total_white_s = gp.quicksum((x[s, n, 'White'] for n in neighborhoods))
        total_students_s = gp.quicksum((x[s, n, g] for n in neighborhoods for g in groups))
        m.addConstr(total_white_s >= 0.5 * total_students_s, name=f'racial_lb_{s}')
        m.addConstr(total_white_s <= 0.7 * total_students_s, name=f'racial_ub_{s}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')