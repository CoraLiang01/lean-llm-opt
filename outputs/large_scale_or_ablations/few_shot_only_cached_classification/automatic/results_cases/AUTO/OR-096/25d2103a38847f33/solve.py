import gurobipy as gp
import pandas as pd
import numpy as np
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
dist_df['School'] = dist_df['School'].astype(str).str.strip()
distance = {}
for _, row in dist_df.iterrows():
    s = str(row['School']).strip()
    for n in neighs:
        distance[s, n] = float(row[n])
total_white = sum((pop_white[n] for n in neighs))
total_nonwhite = sum((pop_nonwhite[n] for n in neighs))
total_students = total_white + total_nonwhite
district_white_ratio = total_white / total_students if total_students > 0 else 0.0
min_white_ratio = 0.5
max_white_ratio = 0.7
m = gp.Model('SchoolAssignment')
x = m.addVars(schools, neighs, groups, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((distance[s, n] * x[s, n, g] for s in schools for n in neighs for g in groups)), gp.GRB.MINIMIZE)
for n in neighs:
    m.addConstr(gp.quicksum((x[s, n, 'White'] for s in schools)) == pop_white[n], name=f'assign_white_{n}')
    m.addConstr(gp.quicksum((x[s, n, 'NonWhite'] for s in schools)) == pop_nonwhite[n], name=f'assign_nonwhite_{n}')
for s in schools:
    m.addConstr(gp.quicksum((x[s, n, g] for n in neighs for g in groups)) <= school_cap[s], name=f'capacity_{s}')
for s in schools:
    total_white_s = gp.quicksum((x[s, n, 'White'] for n in neighs))
    total_students_s = gp.quicksum((x[s, n, g] for n in neighs for g in groups))
    m.addConstr(total_white_s >= min_white_ratio * total_students_s, name=f'racial_min_{s}')
    m.addConstr(total_white_s <= max_white_ratio * total_students_s, name=f'racial_max_{s}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total travel distance: {m.objVal:.2f} miles')
    print('\nAssignment by school:')
    for s in schools:
        total_white_s = sum((x[s, n, 'White'].X for n in neighs))
        total_nonwhite_s = sum((x[s, n, 'NonWhite'].X for n in neighs))
        total_s = total_white_s + total_nonwhite_s
        white_pct = total_white_s / total_s * 100 if total_s > 0 else 0.0
        print(f'  School {s}:')
        print(f'    Total assigned: {int(round(total_s))} (Capacity: {school_cap[s]})')
        print(f'    White: {int(round(total_white_s))}, NonWhite: {int(round(total_nonwhite_s))}, White %: {white_pct:.2f}%')
    print('\nSample assignment details (nonzero only):')
    for n in neighs:
        for s in schools:
            for g in groups:
                val = x[s, n, g].X
                if val > 1e-05:
                    print(f'  {int(round(val))} {g} students from {n} assigned to School {s}')
else:
    print(f'No optimal solution found. Status: {m.status}')