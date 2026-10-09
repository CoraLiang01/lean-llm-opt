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
population = {}
for n in neighborhoods:
    population[n] = {'White': pop_white[n], 'NonWhite': pop_nonwhite[n]}
distance_df['School'] = distance_df['School'].str.strip()
distance = {}
for (_, row) in distance_df.iterrows():
    s = row['School']
    distance[s] = {}
    for n in neighborhoods:
        val = row[n]
        try:
            distance[s][n] = float(val)
        except Exception:
            raise ValueError(f"Missing or invalid distance for school {s}, neighborhood {n}: '{val}'")
total_white = sum((pop_white[n] for n in neighborhoods))
total_nonwhite = sum((pop_nonwhite[n] for n in neighborhoods))
total_students = total_white + total_nonwhite
district_white_ratio = total_white / total_students if total_students > 0 else 0.0
m = gp.Model('SchoolAssignment')
x_vars = m.addVars(schools, neighborhoods, groups, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((distance[s][n] * x_vars[s, n, g] for s in schools for n in neighborhoods for g in groups)), gp.GRB.MINIMIZE)
for n in neighborhoods:
    for g in groups:
        m.addConstr(gp.quicksum((x_vars[s, n, g] for s in schools)) == population[n][g])
for s in schools:
    m.addConstr(gp.quicksum((x_vars[s, n, g] for n in neighborhoods for g in groups)) <= capacity[s])
for s in schools:
    total_white_s = gp.quicksum((x_vars[s, n, 'White'] for n in neighborhoods))
    total_students_s = gp.quicksum((x_vars[s, n, g] for n in neighborhoods for g in groups))
    m.addConstr(total_white_s >= 0.5 * total_students_s)
    m.addConstr(total_white_s <= 0.7 * total_students_s)
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
        total_white_s = sum((x_vars[s, n, 'White'].X for n in neighborhoods))
        total_nonwhite_s = sum((x_vars[s, n, 'NonWhite'].X for n in neighborhoods))
        total_s = total_white_s + total_nonwhite_s
        pct_white = total_white_s / total_s * 100 if total_s > 0 else 0.0
        print(f'School {s}: Total={total_s:.1f}, White={total_white_s:.1f}, NonWhite={total_nonwhite_s:.1f}, %White={pct_white:.1f}% (Capacity={capacity[s]})')
else:
    print(f'No optimal solution found. Status: {m.status}')