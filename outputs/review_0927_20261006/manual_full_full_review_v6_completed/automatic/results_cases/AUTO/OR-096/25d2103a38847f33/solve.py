import gurobipy as gp
import pandas as pd
import numpy as np
school_capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/school_capacity.csv', dtype=str, keep_default_na=False)
school_capacity_df['School'] = school_capacity_df['School'].str.strip()
school_capacity_df['Capacity'] = school_capacity_df['Capacity'].astype(int)
schools = list(school_capacity_df['School'])
capacity_dict = dict(zip(school_capacity_df['School'], school_capacity_df['Capacity']))
neigh_pop_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/neighborhoods_population.csv', dtype=str, keep_default_na=False)
neigh_pop_df['Neighborhood'] = neigh_pop_df['Neighborhood'].str.strip()
neigh_pop_df['Population_White'] = neigh_pop_df['Population_White'].astype(int)
neigh_pop_df['Population_NonWhite'] = neigh_pop_df['Population_NonWhite'].astype(int)
neighborhoods = list(neigh_pop_df['Neighborhood'])
pop_white_dict = dict(zip(neigh_pop_df['Neighborhood'], neigh_pop_df['Population_White']))
pop_nonwhite_dict = dict(zip(neigh_pop_df['Neighborhood'], neigh_pop_df['Population_NonWhite']))
distance_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/distance.csv', dtype=str, keep_default_na=False)
distance_df['School'] = distance_df['School'].str.strip()
for n in neighborhoods:
    distance_df[n] = distance_df[n].astype(float)
distance_dict = {}
for _, row in distance_df.iterrows():
    s = row['School']
    for n in neighborhoods:
        distance_dict[s, n] = float(row[n])
groups = ['White', 'NonWhite']
m = gp.Model('SchoolAssignment')
x_vars = m.addVars(schools, neighborhoods, groups, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((distance_dict[s, n] * x_vars[s, n, g] for s in schools for n in neighborhoods for g in groups)), gp.GRB.MINIMIZE)
for n in neighborhoods:
    m.addConstr(gp.quicksum((x_vars[s, n, 'White'] for s in schools)) == pop_white_dict[n])
    m.addConstr(gp.quicksum((x_vars[s, n, 'NonWhite'] for s in schools)) == pop_nonwhite_dict[n])
for s in schools:
    m.addConstr(gp.quicksum((x_vars[s, n, g] for n in neighborhoods for g in groups)) <= capacity_dict[s])
district_white_ratio = 0.6
tolerance = 0.1
min_white_ratio = district_white_ratio - tolerance
max_white_ratio = district_white_ratio + tolerance
for s in schools:
    total_white = gp.quicksum((x_vars[s, n, 'White'] for n in neighborhoods))
    total_students = gp.quicksum((x_vars[s, n, g] for n in neighborhoods for g in groups))
    m.addConstr(total_white >= min_white_ratio * total_students)
    m.addConstr(total_white <= max_white_ratio * total_students)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total distance: {m.objVal:.2f} miles')
    print('\nAssignment by school, neighborhood, and group (nonzero only):')
    for s in schools:
        for n in neighborhoods:
            for g in groups:
                val = x_vars[s, n, g].X
                if val > 1e-05:
                    print(f'School {s}, Neighborhood {n}, Group {g}: {val:.2f} students')
    print('\nSchool-level summary:')
    for s in schools:
        total_white = sum((x_vars[s, n, 'White'].X for n in neighborhoods))
        total_nonwhite = sum((x_vars[s, n, 'NonWhite'].X for n in neighborhoods))
        total_students = total_white + total_nonwhite
        white_pct = total_white / total_students * 100 if total_students > 0 else 0.0
        print(f'School {s}: {total_students:.0f} students (White: {total_white:.0f}, NonWhite: {total_nonwhite:.0f}, White %: {white_pct:.2f}%)')
else:
    print(f'No optimal solution found. Status: {m.status}')