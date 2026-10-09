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
school_capacities = dict(zip(school_capacity_df['School'], school_capacity_df['Capacity']))
neighborhoods_population_df['Population_White'] = neighborhoods_population_df['Population_White'].astype(int)
neighborhoods_population_df['Population_NonWhite'] = neighborhoods_population_df['Population_NonWhite'].astype(int)
pop_white = neighborhoods_population_df.set_index('Neighborhood')['Population_White'].to_dict()
pop_nonwhite = neighborhoods_population_df.set_index('Neighborhood')['Population_NonWhite'].to_dict()
pop_by_group = {'White': pop_white, 'NonWhite': pop_nonwhite}
total_white = sum((pop_white[n] for n in neighborhoods))
total_nonwhite = sum((pop_nonwhite[n] for n in neighborhoods))
total_students = total_white + total_nonwhite
district_white_ratio = total_white / total_students if total_students > 0 else 0.0
distance_df['School'] = distance_df['School'].str.strip()
distance_df = distance_df.set_index('School')
for s in schools:
    if s not in distance_df.index:
        raise ValueError(f"School '{s}' missing from distance.csv")
for n in neighborhoods:
    if n not in distance_df.columns:
        raise ValueError(f"Neighborhood '{n}' missing from distance.csv columns")
distance_matrix = {}
for s in schools:
    distance_matrix[s] = {}
    for n in neighborhoods:
        val = distance_df.loc[s, n]
        try:
            distance_matrix[s][n] = float(val)
        except Exception:
            raise ValueError(f"Distance value for school '{s}', neighborhood '{n}' is not a valid float: '{val}'")
m = gp.Model('SchoolAssignment')
x_vars = m.addVars(schools, neighborhoods, groups, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((x_vars[s, n, g] * distance_matrix[s][n] for s in schools for n in neighborhoods for g in groups)), gp.GRB.MINIMIZE)
for n in neighborhoods:
    for g in groups:
        pop = pop_by_group[g][n]
        m.addConstr(gp.quicksum((x_vars[s, n, g] for s in schools)) == pop)
for s in schools:
    m.addConstr(gp.quicksum((x_vars[s, n, g] for n in neighborhoods for g in groups)) <= school_capacities[s])
for s in schools:
    total_white_in_s = gp.quicksum((x_vars[s, n, 'White'] for n in neighborhoods))
    total_in_s = gp.quicksum((x_vars[s, n, g] for n in neighborhoods for g in groups))
    m.addConstr(total_white_in_s >= 0.5 * total_in_s)
    m.addConstr(total_white_in_s <= 0.7 * total_in_s)
m.optimize()