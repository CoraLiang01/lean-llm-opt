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
schools = school_capacity_df['School'].astype(str).tolist()
neighborhoods = neighborhoods_population_df['Neighborhood'].astype(str).tolist()
groups = ['White', 'NonWhite']
school_capacity_df['Capacity'] = school_capacity_df['Capacity'].astype(float)
school_capacities = dict(zip(school_capacity_df['School'].astype(str), school_capacity_df['Capacity']))
neighborhoods_population_df['Population_White'] = neighborhoods_population_df['Population_White'].astype(float)
neighborhoods_population_df['Population_NonWhite'] = neighborhoods_population_df['Population_NonWhite'].astype(float)
pop_white = dict(zip(neighborhoods_population_df['Neighborhood'].astype(str), neighborhoods_population_df['Population_White']))
pop_nonwhite = dict(zip(neighborhoods_population_df['Neighborhood'].astype(str), neighborhoods_population_df['Population_NonWhite']))
pop_by_group = {}
for n in neighborhoods:
    pop_by_group[n] = {'White': pop_white[n], 'NonWhite': pop_nonwhite[n]}
distance_dict = {}
for (_, row) in distance_df.iterrows():
    school = str(row['School'])
    distance_dict[school] = {}
    for n in neighborhoods:
        val = row[n]
        try:
            distance_dict[school][n] = float(val)
        except Exception:
            raise ValueError(f"Distance value for school {school}, neighborhood {n} is not a valid float: '{val}'")
total_white = sum((pop_white[n] for n in neighborhoods))
total_nonwhite = sum((pop_nonwhite[n] for n in neighborhoods))
total_students = total_white + total_nonwhite
district_white_ratio = 0.6
m = gp.Model('SchoolAssignment')
x_vars = m.addVars(schools, neighborhoods, groups, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((distance_dict[s][n] * x_vars[s, n, g] for s in schools for n in neighborhoods for g in groups)), gp.GRB.MINIMIZE)
for n in neighborhoods:
    for g in groups:
        m.addConstr(gp.quicksum((x_vars[s, n, g] for s in schools)) == pop_by_group[n][g], name=f'Assign_{n}_{g}')
for s in schools:
    m.addConstr(gp.quicksum((x_vars[s, n, g] for n in neighborhoods for g in groups)) <= school_capacities[s], name=f'Capacity_{s}')
for s in schools:
    total_white_s = gp.quicksum((x_vars[s, n, 'White'] for n in neighborhoods))
    total_nonwhite_s = gp.quicksum((x_vars[s, n, 'NonWhite'] for n in neighborhoods))
    total_students_s = total_white_s + total_nonwhite_s
    m.addConstr(total_white_s >= 0.5 * total_students_s, name=f'RacialBalanceLower_{s}')
    m.addConstr(total_white_s <= 0.7 * total_students_s, name=f'RacialBalanceUpper_{s}')
m.optimize()