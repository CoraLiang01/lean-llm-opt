import gurobipy as gp
import pandas as pd
import numpy as np
school_capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/school_capacity.csv', dtype=str, keep_default_na=False)
school_capacity_df['School'] = school_capacity_df['School'].str.strip()
school_capacity_df['Capacity'] = school_capacity_df['Capacity'].astype(int)
schools = list(school_capacity_df['School'])
school_capacities = dict(zip(school_capacity_df['School'], school_capacity_df['Capacity']))
neigh_pop_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/neighborhoods_population.csv', dtype=str, keep_default_na=False)
neigh_pop_df['Neighborhood'] = neigh_pop_df['Neighborhood'].str.strip()
neigh_pop_df['Population_White'] = neigh_pop_df['Population_White'].astype(int)
neigh_pop_df['Population_NonWhite'] = neigh_pop_df['Population_NonWhite'].astype(int)
neighborhoods = list(neigh_pop_df['Neighborhood'])
pop_white = dict(zip(neigh_pop_df['Neighborhood'], neigh_pop_df['Population_White']))
pop_nonwhite = dict(zip(neigh_pop_df['Neighborhood'], neigh_pop_df['Population_NonWhite']))
distance_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/distance.csv', dtype=str, keep_default_na=False)
distance_df['School'] = distance_df['School'].str.strip()
distance_cols = [col for col in distance_df.columns if col != 'School']
distance_cols = [col.strip() for col in distance_cols]
distance = {}
for idx, row in distance_df.iterrows():
    school = row['School']
    for n_col in distance_cols:
        n_id = n_col.strip()
        distance[school, n_id] = float(row[n_col])
groups = ['White', 'NonWhite']
total_white = sum((pop_white[n] for n in neighborhoods))
total_nonwhite = sum((pop_nonwhite[n] for n in neighborhoods))
total_students = total_white + total_nonwhite
district_white_ratio = total_white / total_students if total_students > 0 else 0.0
m = gp.Model('SchoolAssignment')
x_vars = m.addVars(schools, neighborhoods, groups, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((distance[s, n] * x_vars[s, n, g] for s in schools for n in neighborhoods for g in groups)), gp.GRB.MINIMIZE)
for n in neighborhoods:
    m.addConstr(gp.quicksum((x_vars[s, n, 'White'] for s in schools)) == pop_white[n], name=f'assign_white_{n}')
    m.addConstr(gp.quicksum((x_vars[s, n, 'NonWhite'] for s in schools)) == pop_nonwhite[n], name=f'assign_nonwhite_{n}')
for s in schools:
    m.addConstr(gp.quicksum((x_vars[s, n, g] for n in neighborhoods for g in groups)) <= school_capacities[s], name=f'capacity_{s}')
for s in schools:
    total_white_s = gp.quicksum((x_vars[s, n, 'White'] for n in neighborhoods))
    total_nonwhite_s = gp.quicksum((x_vars[s, n, 'NonWhite'] for n in neighborhoods))
    total_s = total_white_s + total_nonwhite_s
    m.addConstr(total_white_s >= 0.5 * total_s, name=f'racial_lb_{s}')
    m.addConstr(total_white_s <= 0.7 * total_s, name=f'racial_ub_{s}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total travel distance: {m.objVal:.4f} miles')
    print('\nAssignment by school:')
    for s in schools:
        total_white_s = sum((x_vars[s, n, 'White'].X for n in neighborhoods))
        total_nonwhite_s = sum((x_vars[s, n, 'NonWhite'].X for n in neighborhoods))
        total_s = total_white_s + total_nonwhite_s
        white_pct = total_white_s / total_s if total_s > 0 else 0.0
        print(f'School {s}:')
        print(f'  Total assigned: {int(round(total_s))} (Capacity: {school_capacities[s]})')
        print(f'    White: {int(round(total_white_s))} ({white_pct * 100:.2f}%)')
        print(f'    NonWhite: {int(round(total_nonwhite_s))} ({100 - white_pct * 100:.2f}%)')
    print('\nSample assignment (nonzero only):')
    for s in schools:
        for n in neighborhoods:
            for g in groups:
                val = x_vars[s, n, g].X
                if val > 1e-05:
                    print(f'  {val:.2f} {g} students from {n} assigned to {s}')
else:
    print(f'No optimal solution found. Status: {m.status}')