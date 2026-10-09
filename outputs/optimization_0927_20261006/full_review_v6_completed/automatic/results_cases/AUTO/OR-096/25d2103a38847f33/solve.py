import gurobipy as gp
import pandas as pd
import numpy as np
school_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/school_capacity.csv'
neighborhoods_population_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/neighborhoods_population.csv'
distance_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/distance.csv'
school_capacity_df = pd.read_csv(school_capacity_path, dtype=str, keep_default_na=False)
neighborhoods_population_df = pd.read_csv(neighborhoods_population_path, dtype=str, keep_default_na=False)
distance_df = pd.read_csv(distance_path, dtype=str, keep_default_na=False)
school_capacity_df['School'] = school_capacity_df['School'].str.strip()
schools = list(school_capacity_df['School'])
neighborhoods_population_df['Neighborhood'] = neighborhoods_population_df['Neighborhood'].str.strip()
neighborhoods = list(neighborhoods_population_df['Neighborhood'])
groups = ['White', 'NonWhite']
school_capacity_df['Capacity'] = school_capacity_df['Capacity'].astype(int)
capacity = dict(zip(school_capacity_df['School'], school_capacity_df['Capacity']))
neighborhoods_population_df['Population_White'] = neighborhoods_population_df['Population_White'].astype(int)
neighborhoods_population_df['Population_NonWhite'] = neighborhoods_population_df['Population_NonWhite'].astype(int)
pop_white = dict(zip(neighborhoods_population_df['Neighborhood'], neighborhoods_population_df['Population_White']))
pop_nonwhite = dict(zip(neighborhoods_population_df['Neighborhood'], neighborhoods_population_df['Population_NonWhite']))
distance_df['School'] = distance_df['School'].str.strip()
distance_dict = {}
for (_, row) in distance_df.iterrows():
    s = row['School']
    for n in neighborhoods:
        val = float(row[n])
        distance_dict[s, n] = val
total_white = sum((pop_white[n] for n in neighborhoods))
total_nonwhite = sum((pop_nonwhite[n] for n in neighborhoods))
total_students = total_white + total_nonwhite
district_white_ratio = total_white / total_students if total_students > 0 else 0.0
m = gp.Model('SchoolAssignment')
x_vars = m.addVars(schools, neighborhoods, groups, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((distance_dict[s, n] * x_vars[s, n, g] for s in schools for n in neighborhoods for g in groups)), gp.GRB.MINIMIZE)
for n in neighborhoods:
    m.addConstr(gp.quicksum((x_vars[s, n, 'White'] for s in schools)) == pop_white[n], name=f'Assign_White_{n}')
    m.addConstr(gp.quicksum((x_vars[s, n, 'NonWhite'] for s in schools)) == pop_nonwhite[n], name=f'Assign_NonWhite_{n}')
for s in schools:
    m.addConstr(gp.quicksum((x_vars[s, n, g] for n in neighborhoods for g in groups)) <= capacity[s], name=f'Capacity_{s}')
for s in schools:
    total_white_s = gp.quicksum((x_vars[s, n, 'White'] for n in neighborhoods))
    total_nonwhite_s = gp.quicksum((x_vars[s, n, 'NonWhite'] for n in neighborhoods))
    total_s = total_white_s + total_nonwhite_s
    m.addConstr(total_white_s >= 0.5 * total_s, name=f'RacialBalance_LB_{s}')
    m.addConstr(total_white_s <= 0.7 * total_s, name=f'RacialBalance_UB_{s}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total student-miles: {m.objVal:.2f}')
    print('\n--- Assignment by School ---')
    for s in schools:
        total_white_s = sum((x_vars[s, n, 'White'].X for n in neighborhoods))
        total_nonwhite_s = sum((x_vars[s, n, 'NonWhite'].X for n in neighborhoods))
        total_s = total_white_s + total_nonwhite_s
        white_pct = total_white_s / total_s * 100 if total_s > 0 else 0.0
        print(f'School {s}:')
        print(f'  Total assigned: {int(round(total_s))} (Capacity: {capacity[s]})')
        print(f'    White:    {int(round(total_white_s))}')
        print(f'    NonWhite: {int(round(total_nonwhite_s))}')
        print(f'    White %:  {white_pct:.2f}%')
    print('\n--- Assignment by Neighborhood ---')
    for n in neighborhoods:
        for g in groups:
            for s in schools:
                val = x_vars[s, n, g].X
                if val > 1e-05:
                    print(f'Neighborhood {n}, Group {g}, School {s}: {int(round(val))}')
else:
    print(f'No optimal solution found. Status: {m.status}')