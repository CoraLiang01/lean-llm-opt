import gurobipy as gp
import pandas as pd
import numpy as np
import re
school_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/school_capacity.csv'
neighborhoods_population_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/neighborhoods_population.csv'
distance_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/distance.csv'
school_capacity_df = pd.read_csv(school_capacity_path, sep=',', dtype=str, keep_default_na=False)
if 'School' not in school_capacity_df.columns or 'Capacity' not in school_capacity_df.columns:
    raise KeyError("school_capacity.csv must have columns 'School' and 'Capacity'")
school_capacity_df['Capacity'] = school_capacity_df['Capacity'].astype(float)
schools = list(school_capacity_df['School'])
school_capacities = dict(zip(school_capacity_df['School'], school_capacity_df['Capacity']))
neigh_pop_df = pd.read_csv(neighborhoods_population_path, sep=',', dtype=str, keep_default_na=False)
if 'Neighborhood' not in neigh_pop_df.columns or 'Population_White' not in neigh_pop_df.columns or 'Population_NonWhite' not in neigh_pop_df.columns:
    raise KeyError("neighborhoods_population.csv must have columns 'Neighborhood', 'Population_White', 'Population_NonWhite'")
neigh_pop_df['Population_White'] = neigh_pop_df['Population_White'].astype(float)
neigh_pop_df['Population_NonWhite'] = neigh_pop_df['Population_NonWhite'].astype(float)
neighborhoods = list(neigh_pop_df['Neighborhood'])
groups = ['White', 'NonWhite']
pop = {}
for _, row in neigh_pop_df.iterrows():
    n = row['Neighborhood']
    pop[n] = {'White': row['Population_White'], 'NonWhite': row['Population_NonWhite']}
total_white = neigh_pop_df['Population_White'].sum()
total_nonwhite = neigh_pop_df['Population_NonWhite'].sum()
total_students = total_white + total_nonwhite
if total_students == 0:
    raise ValueError('Total number of students in the district is zero.')
district_white_ratio = total_white / total_students
distance_df = pd.read_csv(distance_path, sep=',', dtype=str, keep_default_na=False)
distance_df['School'] = distance_df['School'].astype(str)
distance_df.set_index('School', inplace=True)
distance_df.columns = [str(col) for col in distance_df.columns]
for s in schools:
    if s not in distance_df.index:
        raise KeyError(f"School '{s}' from school_capacity.csv not found in distance.csv")
for n in neighborhoods:
    if n not in distance_df.columns:
        raise KeyError(f"Neighborhood '{n}' from neighborhoods_population.csv not found in distance.csv columns")
distance = {}
for s in schools:
    distance[s] = {}
    for n in neighborhoods:
        val = distance_df.loc[s, n]
        try:
            distance[s][n] = float(val)
        except Exception:
            raise ValueError(f"Distance value for school '{s}', neighborhood '{n}' is not a valid number: '{val}'")
m = gp.Model('SchoolAssignment')
x_vars = m.addVars(schools, neighborhoods, groups, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((distance[s][n] * x_vars[s, n, g] for s in schools for n in neighborhoods for g in groups)), gp.GRB.MINIMIZE)
for n in neighborhoods:
    for g in groups:
        m.addConstr(gp.quicksum((x_vars[s, n, g] for s in schools)) == pop[n][g], name=f'assign_{n}_{g}')
for s in schools:
    m.addConstr(gp.quicksum((x_vars[s, n, g] for n in neighborhoods for g in groups)) <= school_capacities[s], name=f'capacity_{s}')
for s in schools:
    total_white_expr = gp.quicksum((x_vars[s, n, 'White'] for n in neighborhoods))
    total_students_expr = gp.quicksum((x_vars[s, n, g] for n in neighborhoods for g in groups))
    m.addGenConstrIndicator(m.addVar(vtype=gp.GRB.BINARY, name=f'has_students_{s}'), True, total_white_expr >= 0.5 * total_students_expr, name=f'racial_lb_{s}')
    m.addGenConstrIndicator(m.getVarByName(f'has_students_{s}'), True, total_white_expr <= 0.7 * total_students_expr, name=f'racial_ub_{s}')
    m.addGenConstrIndicator(m.getVarByName(f'has_students_{s}'), True, total_students_expr >= 1e-05, name=f'has_students_lb_{s}')
    m.addGenConstrIndicator(m.getVarByName(f'has_students_{s}'), False, total_students_expr <= 1e-06, name=f'has_students_ub_{s}')
m.optimize()