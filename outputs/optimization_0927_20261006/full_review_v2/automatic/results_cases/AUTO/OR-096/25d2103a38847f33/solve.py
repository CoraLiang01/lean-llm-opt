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

def norm_str(x):
    return x.strip().casefold()
school_capacity_df['School_norm'] = school_capacity_df['School'].apply(norm_str)
distance_df['School_norm'] = distance_df['School'].apply(norm_str)
neighborhoods_population_df['Neighborhood_norm'] = neighborhoods_population_df['Neighborhood'].apply(norm_str)
schools = list(school_capacity_df['School'])
schools_norm = [norm_str(s) for s in schools]
school_id_map = dict(zip(schools_norm, schools))
neighborhoods = list(neighborhoods_population_df['Neighborhood'])
neighborhoods_norm = [norm_str(n) for n in neighborhoods]
neigh_id_map = dict(zip(neighborhoods_norm, neighborhoods))
groups = ['White', 'NonWhite']
school_capacity = {}
for (idx, row) in school_capacity_df.iterrows():
    s_norm = row['School_norm']
    cap = int(row['Capacity'])
    school_capacity[s_norm] = cap
pop_white = {}
pop_nonwhite = {}
for (idx, row) in neighborhoods_population_df.iterrows():
    n_norm = row['Neighborhood_norm']
    pop_white[n_norm] = int(row['Population_White'])
    pop_nonwhite[n_norm] = int(row['Population_NonWhite'])
distance = {}
for (idx, row) in distance_df.iterrows():
    s_norm = row['School_norm']
    for n in neighborhoods:
        n_norm = norm_str(n)
        val = row[n]
        try:
            distance[s_norm, n_norm] = float(val)
        except Exception:
            raise ValueError(f"Distance missing or invalid for school '{row['School']}' and neighborhood '{n}'")
total_white = sum((pop_white[n] for n in neighborhoods_norm))
total_nonwhite = sum((pop_nonwhite[n] for n in neighborhoods_norm))
total_students = total_white + total_nonwhite
district_white_ratio = total_white / total_students if total_students > 0 else 0.0
m = gp.Model('SchoolAssignment')
x_vars = m.addVars(schools_norm, neighborhoods_norm, groups, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((distance[s, n] * x_vars[s, n, g] for s in schools_norm for n in neighborhoods_norm for g in groups)), gp.GRB.MINIMIZE)
for n in neighborhoods_norm:
    m.addConstr(gp.quicksum((x_vars[s, n, 'White'] for s in schools_norm)) == pop_white[n], name=f'assign_white_{n}')
    m.addConstr(gp.quicksum((x_vars[s, n, 'NonWhite'] for s in schools_norm)) == pop_nonwhite[n], name=f'assign_nonwhite_{n}')
for s in schools_norm:
    m.addConstr(gp.quicksum((x_vars[s, n, g] for n in neighborhoods_norm for g in groups)) <= school_capacity[s], name=f'capacity_{s}')
for s in schools_norm:
    total_white_s = gp.quicksum((x_vars[s, n, 'White'] for n in neighborhoods_norm))
    total_students_s = gp.quicksum((x_vars[s, n, g] for n in neighborhoods_norm for g in groups))
    m.addConstr(total_white_s >= 0.5 * total_students_s, name=f'racial_lb_{s}')
    m.addConstr(total_white_s <= 0.7 * total_students_s, name=f'racial_ub_{s}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total travel distance: {m.objVal:.2f} miles')
    print('\nAssignment by school, neighborhood, and group (nonzero only):')
    for s in schools_norm:
        s_orig = school_id_map[s]
        for n in neighborhoods_norm:
            n_orig = neigh_id_map[n]
            for g in groups:
                val = x_vars[s, n, g].X
                if val > 1e-05:
                    print(f'School {s_orig}, Neighborhood {n_orig}, Group {g}: {val:.2f} students')
    print('\nSchool-level summary:')
    for s in schools_norm:
        s_orig = school_id_map[s]
        total_white_s = sum((x_vars[s, n, 'White'].X for n in neighborhoods_norm))
        total_nonwhite_s = sum((x_vars[s, n, 'NonWhite'].X for n in neighborhoods_norm))
        total_s = total_white_s + total_nonwhite_s
        white_pct = total_white_s / total_s * 100 if total_s > 0 else 0.0
        print(f'School {s_orig}: Total assigned = {total_s:.1f}, White = {total_white_s:.1f}, NonWhite = {total_nonwhite_s:.1f}, White % = {white_pct:.2f}% (Capacity: {school_capacity[s]})')
else:
    print(f'No optimal solution found. Status: {m.status}')