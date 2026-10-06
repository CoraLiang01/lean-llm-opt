import gurobipy as gp
import pandas as pd
import numpy as np
school_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/school_capacity.csv'
neigh_pop_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/neighborhoods_population.csv'
distance_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/distance.csv'
school_df = pd.read_csv(school_capacity_path, sep=',')
school_df['School'] = school_df['School'].astype(str).str.strip()
schools = list(school_df['School'])
school_cap = dict(zip(school_df['School'], school_df['Capacity']))
neigh_df = pd.read_csv(neigh_pop_path, sep=',')
neigh_df['Neighborhood'] = neigh_df['Neighborhood'].astype(str).str.strip()
neighborhoods = list(neigh_df['Neighborhood'])
pop_white = dict(zip(neigh_df['Neighborhood'], neigh_df['Population_White']))
pop_nonwhite = dict(zip(neigh_df['Neighborhood'], neigh_df['Population_NonWhite']))
groups = ['White', 'NonWhite']
dist_df = pd.read_csv(distance_path, sep=',')
dist_df['School'] = dist_df['School'].astype(str).str.strip()
for n in neighborhoods:
    if n not in dist_df.columns:
        raise ValueError(f'Neighborhood {n} missing in distance.csv columns')
distance = {}
for _, row in dist_df.iterrows():
    s = str(row['School']).strip()
    for n in neighborhoods:
        distance[s, n] = float(row[n])
if set(schools) != set(dist_df['School'].unique()):
    raise ValueError('Mismatch between schools in school_capacity.csv and distance.csv')
if set(neighborhoods) != set(neigh_df['Neighborhood'].unique()):
    raise ValueError('Mismatch between neighborhoods in neighborhoods_population.csv and distance.csv columns')
m = gp.Model('SchoolAssignment')
x = m.addVars(schools, neighborhoods, groups, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((x[s, n, g] * distance[s, n] for s in schools for n in neighborhoods for g in groups)), gp.GRB.MINIMIZE)
for n in neighborhoods:
    m.addConstr(gp.quicksum((x[s, n, 'White'] for s in schools)) == pop_white[n])
    m.addConstr(gp.quicksum((x[s, n, 'NonWhite'] for s in schools)) == pop_nonwhite[n])
for s in schools:
    m.addConstr(gp.quicksum((x[s, n, g] for n in neighborhoods for g in groups)) <= school_cap[s])
for s in schools:
    total_white = gp.quicksum((x[s, n, 'White'] for n in neighborhoods))
    total_nonwhite = gp.quicksum((x[s, n, 'NonWhite'] for n in neighborhoods))
    total_students = total_white + total_nonwhite
    m.addConstr(total_white >= 0.5 * total_students)
    m.addConstr(total_white <= 0.7 * total_students)
m.optimize()