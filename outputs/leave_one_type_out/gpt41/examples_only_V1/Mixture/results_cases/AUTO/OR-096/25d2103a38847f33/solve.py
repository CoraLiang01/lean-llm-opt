import gurobipy as gp
import pandas as pd
import numpy as np
import re
school_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/school_capacity.csv'
neigh_pop_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/neighborhoods_population.csv'
distance_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/distance.csv'
school_df = pd.read_csv(school_capacity_path, sep=',')
schools = [str(s).strip() for s in school_df['School']]
school_cap = {str(row['School']).strip(): int(row['Capacity']) for _, row in school_df.iterrows()}
neigh_df = pd.read_csv(neigh_pop_path, sep=',')
neighs = [str(n).strip() for n in neigh_df['Neighborhood']]
pop_white = {str(row['Neighborhood']).strip(): int(row['Population_White']) for _, row in neigh_df.iterrows()}
pop_nonwhite = {str(row['Neighborhood']).strip(): int(row['Population_NonWhite']) for _, row in neigh_df.iterrows()}
groups = ['White', 'NonWhite']
dist_df = pd.read_csv(distance_path, sep=',')
dist_df['School'] = dist_df['School'].apply(lambda x: str(x).strip())
distance = {}
for _, row in dist_df.iterrows():
    s = str(row['School']).strip()
    distance[s] = {}
    for n in neighs:
        distance[s][n] = float(row[n])
if set(schools) != set(distance.keys()):
    raise ValueError('Mismatch between schools in school_capacity.csv and distance.csv')
if not all((n in distance[schools[0]] for n in neighs)):
    raise ValueError('Not all neighborhoods in neighborhoods_population.csv are present in distance.csv columns')
if not all((n in pop_white and n in pop_nonwhite for n in neighs)):
    raise ValueError('Neighborhoods missing in population data')
total_white = sum((pop_white[n] for n in neighs))
total_nonwhite = sum((pop_nonwhite[n] for n in neighs))
total_students = total_white + total_nonwhite
district_white_ratio = total_white / total_students if total_students > 0 else 0.0
district_nonwhite_ratio = total_nonwhite / total_students if total_students > 0 else 0.0
lower_white = 0.6 - 0.1
upper_white = 0.6 + 0.1
m = gp.Model('SchoolAssignment')
x = m.addVars(schools, neighs, groups, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((x[s, n, g] * distance[s][n] for s in schools for n in neighs for g in groups)), gp.GRB.MINIMIZE)
for n in neighs:
    m.addConstr(gp.quicksum((x[s, n, 'White'] for s in schools)) == pop_white[n], name='')
    m.addConstr(gp.quicksum((x[s, n, 'NonWhite'] for s in schools)) == pop_nonwhite[n], name='')
for s in schools:
    m.addConstr(gp.quicksum((x[s, n, g] for n in neighs for g in groups)) <= school_cap[s], name='')
for s in schools:
    W_s = gp.quicksum((x[s, n, 'White'] for n in neighs))
    T_s = gp.quicksum((x[s, n, g] for n in neighs for g in groups))
    y_s = m.addVar(vtype=gp.GRB.BINARY, name=f'y_{s}')
    epsilon = 0.0001
    bigM = sum((pop_white[n] + pop_nonwhite[n] for n in neighs))
    m.addConstr(T_s >= epsilon * y_s, name='')
    m.addConstr(T_s <= bigM * y_s, name='')
    m.addGenConstrIndicator(y_s, True, W_s >= lower_white * T_s, name='')
    m.addGenConstrIndicator(y_s, True, W_s <= upper_white * T_s, name='')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total travel distance: {m.objVal:.4f} miles')
    for s in schools:
        assigned_white = sum((x[s, n, 'White'].X for n in neighs))
        assigned_nonwhite = sum((x[s, n, 'NonWhite'].X for n in neighs))
        total_assigned = assigned_white + assigned_nonwhite
        white_pct = assigned_white / total_assigned * 100 if total_assigned > 0 else 0.0
        print(f'\nSchool {s}:')
        print(f'  Total assigned: {int(round(total_assigned))} (Capacity: {school_cap[s]})')
        print(f'    White:    {int(round(assigned_white))}')
        print(f'    NonWhite: {int(round(assigned_nonwhite))}')
        print(f'    White %:  {white_pct:.2f}%')
    print('\nAssignment by neighborhood (nonzero only):')
    for n in neighs:
        for s in schools:
            for g in groups:
                val = x[s, n, g].X
                if val > 0.0001:
                    print(f'  {val:.2f} {g} students from {n} assigned to School {s}')
else:
    print(f'No optimal solution found. Status: {m.status}')