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
if 'School' not in school_capacity_df.columns:
    raise KeyError("Missing 'School' column in school_capacity.csv")
schools = school_capacity_df['School'].astype(str).tolist()
if 'Neighborhood' not in neighborhoods_population_df.columns:
    raise KeyError("Missing 'Neighborhood' column in neighborhoods_population.csv")
neighborhoods = neighborhoods_population_df['Neighborhood'].astype(str).tolist()
groups = ['White', 'NonWhite']
if 'Capacity' not in school_capacity_df.columns:
    raise KeyError("Missing 'Capacity' column in school_capacity.csv")
school_capacity_dict = {}
for (idx, row) in school_capacity_df.iterrows():
    school = str(row['School'])
    try:
        cap = int(row['Capacity'])
    except Exception:
        raise ValueError(f"Invalid capacity for school {school}: {row['Capacity']}")
    school_capacity_dict[school] = cap
if 'Population_White' not in neighborhoods_population_df.columns or 'Population_NonWhite' not in neighborhoods_population_df.columns:
    raise KeyError('Missing population columns in neighborhoods_population.csv')
pop_white_dict = {}
pop_nonwhite_dict = {}
for (idx, row) in neighborhoods_population_df.iterrows():
    n = str(row['Neighborhood'])
    try:
        pop_white = int(row['Population_White'])
        pop_nonwhite = int(row['Population_NonWhite'])
    except Exception:
        raise ValueError(f"Invalid population for neighborhood {n}: {row['Population_White']}, {row['Population_NonWhite']}")
    pop_white_dict[n] = pop_white
    pop_nonwhite_dict[n] = pop_nonwhite
distance_dict = {}
distance_df = distance_df.set_index('School')
for s in schools:
    if s not in distance_df.index:
        raise KeyError(f'School {s} not found in distance.csv')
    row = distance_df.loc[s]
    distance_dict[s] = {}
    for n in neighborhoods:
        if n not in row.index:
            raise KeyError(f'Neighborhood {n} not found in distance.csv columns')
        try:
            dist = float(row[n])
        except Exception:
            raise ValueError(f'Invalid distance for school {s}, neighborhood {n}: {row[n]}')
        distance_dict[s][n] = dist
total_white = sum((pop_white_dict[n] for n in neighborhoods))
total_nonwhite = sum((pop_nonwhite_dict[n] for n in neighborhoods))
total_students = total_white + total_nonwhite
if total_students == 0:
    raise ValueError('Total number of students in the district is zero.')
district_white_ratio = total_white / total_students
min_white_ratio = 0.5
max_white_ratio = 0.7
m = gp.Model('SchoolAssignment')
x_vars = m.addVars(schools, neighborhoods, groups, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((x_vars[s, n, g] * distance_dict[s][n] for s in schools for n in neighborhoods for g in groups)), gp.GRB.MINIMIZE)
for n in neighborhoods:
    m.addConstr(gp.quicksum((x_vars[s, n, 'White'] for s in schools)) == pop_white_dict[n], name=f'AssignWhite_{n}')
    m.addConstr(gp.quicksum((x_vars[s, n, 'NonWhite'] for s in schools)) == pop_nonwhite_dict[n], name=f'AssignNonWhite_{n}')
for s in schools:
    m.addConstr(gp.quicksum((x_vars[s, n, g] for n in neighborhoods for g in groups)) <= school_capacity_dict[s], name=f'Capacity_{s}')
for s in schools:
    total_white_at_s = gp.quicksum((x_vars[s, n, 'White'] for n in neighborhoods))
    total_nonwhite_at_s = gp.quicksum((x_vars[s, n, 'NonWhite'] for n in neighborhoods))
    total_at_s = total_white_at_s + total_nonwhite_at_s
    m.addConstr(total_white_at_s >= min_white_ratio * (total_white_at_s + total_nonwhite_at_s), name=f'MinWhiteRatio_{s}')
    m.addConstr(total_white_at_s <= max_white_ratio * (total_white_at_s + total_nonwhite_at_s), name=f'MaxWhiteRatio_{s}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total student-miles: {m.objVal:.2f}')
    print('\n--- Assignment Plan (students assigned from each neighborhood to each school) ---')
    for n in neighborhoods:
        for g in groups:
            for s in schools:
                val = x_vars[s, n, g].X
                if val > 1e-05:
                    print(f'Neighborhood {n}, Group {g}, School {s}: {val:.2f} students')
    print('\n--- School Summaries ---')
    for s in schools:
        white = sum((x_vars[s, n, 'White'].X for n in neighborhoods))
        nonwhite = sum((x_vars[s, n, 'NonWhite'].X for n in neighborhoods))
        total = white + nonwhite
        if total > 1e-05:
            pct_white = 100 * white / total
            print(f'School {s}: {total:.2f} students (White: {white:.2f}, NonWhite: {nonwhite:.2f}, %White: {pct_white:.2f}%)')
        else:
            print(f'School {s}: 0 students assigned')
else:
    print(f'No optimal solution found. Status: {m.status}')