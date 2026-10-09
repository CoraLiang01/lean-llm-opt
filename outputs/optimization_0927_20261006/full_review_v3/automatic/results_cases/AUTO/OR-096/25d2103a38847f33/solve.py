import gurobipy as gp
import pandas as pd
import numpy as np
import re
school_capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/school_capacity.csv', dtype=str, keep_default_na=False)
neigh_pop_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/neighborhoods_population.csv', dtype=str, keep_default_na=False)
distance_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/distance.csv', dtype=str, keep_default_na=False)
schools = school_capacity_df['School'].str.strip().tolist()
neighborhoods = neigh_pop_df['Neighborhood'].str.strip().tolist()
groups = ['White', 'NonWhite']
school_capacity_df['Capacity'] = school_capacity_df['Capacity'].astype(int)
school_capacities = dict(zip(school_capacity_df['School'].str.strip(), school_capacity_df['Capacity']))
neigh_pop_df['Population_White'] = neigh_pop_df['Population_White'].astype(int)
neigh_pop_df['Population_NonWhite'] = neigh_pop_df['Population_NonWhite'].astype(int)
pop_white = dict(zip(neigh_pop_df['Neighborhood'].str.strip(), neigh_pop_df['Population_White']))
pop_nonwhite = dict(zip(neigh_pop_df['Neighborhood'].str.strip(), neigh_pop_df['Population_NonWhite']))
pop_by_group = {'White': pop_white, 'NonWhite': pop_nonwhite}
total_white = sum((pop_white[n] for n in neighborhoods))
total_nonwhite = sum((pop_nonwhite[n] for n in neighborhoods))
total_students = total_white + total_nonwhite
district_white_ratio = total_white / total_students if total_students > 0 else 0.0
min_white_ratio = 0.5
max_white_ratio = 0.7
distance_dict = {}
for (idx, row) in distance_df.iterrows():
    school = row['School'].strip()
    for n in neighborhoods:
        val = row[n].strip()
        if val == '':
            raise ValueError(f'Missing distance for school {school}, neighborhood {n}')
        distance_dict[school, n] = float(val)
m = gp.Model('SchoolAssignment')
x_vars = m.addVars(schools, neighborhoods, groups, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((distance_dict[s, n] * x_vars[s, n, g] for s in schools for n in neighborhoods for g in groups)), gp.GRB.MINIMIZE)
for n in neighborhoods:
    for g in groups:
        m.addConstr(gp.quicksum((x_vars[s, n, g] for s in schools)) == pop_by_group[g][n])
for s in schools:
    m.addConstr(gp.quicksum((x_vars[s, n, g] for n in neighborhoods for g in groups)) <= school_capacities[s])
for s in schools:
    total_white_expr = gp.quicksum((x_vars[s, n, 'White'] for n in neighborhoods))
    total_students_expr = gp.quicksum((x_vars[s, n, g] for n in neighborhoods for g in groups))
    m.addConstr(total_white_expr >= min_white_ratio * total_students_expr)
    m.addConstr(total_white_expr <= max_white_ratio * total_students_expr)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total travel distance: {m.objVal:.2f} miles')
    print('\nAssignment by school, neighborhood, group (nonzero only):')
    for s in schools:
        school_total = 0
        school_white = 0
        for n in neighborhoods:
            for g in groups:
                val = x_vars[s, n, g].X
                if val > 1e-05:
                    print(f'  School {s}, Neighborhood {n}, Group {g}: {val:.2f} students')
                school_total += val
                if g == 'White':
                    school_white += val
        if school_total > 0:
            pct_white = 100.0 * school_white / school_total
            print(f'School {s}: Total assigned = {school_total:.2f}, White = {school_white:.2f} ({pct_white:.2f}%)')
        else:
            print(f'School {s}: No students assigned.')
else:
    print(f'No optimal solution found. Status: {m.status}')