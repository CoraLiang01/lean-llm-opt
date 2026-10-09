import gurobipy as gp
import pandas as pd
import numpy as np
school_capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/school_capacity.csv', dtype=str, keep_default_na=False)
neigh_pop_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/neighborhoods_population.csv', dtype=str, keep_default_na=False)
distance_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/distance.csv', dtype=str, keep_default_na=False)
schools = [s.strip() for s in school_capacity_df['School'].tolist()]
if len(schools) != len(set(schools)):
    raise ValueError('Duplicate school identifiers found.')
school_cap = {}
for (_, row) in school_capacity_df.iterrows():
    s = row['School'].strip()
    try:
        cap = int(row['Capacity'])
    except Exception:
        raise ValueError(f"Invalid capacity for school {s}: {row['Capacity']}")
    school_cap[s] = cap
neighs = [n.strip() for n in neigh_pop_df['Neighborhood'].tolist()]
if len(neighs) != len(set(neighs)):
    raise ValueError('Duplicate neighborhood identifiers found.')
groups = ['White', 'NonWhite']
pop_white = {}
pop_nonwhite = {}
for (_, row) in neigh_pop_df.iterrows():
    n = row['Neighborhood'].strip()
    try:
        pop_white[n] = int(row['Population_White'])
        pop_nonwhite[n] = int(row['Population_NonWhite'])
    except Exception:
        raise ValueError(f"Invalid population for neighborhood {n}: {row['Population_White']}, {row['Population_NonWhite']}")
distance_school_ids = [s.strip() for s in distance_df['School'].tolist()]
distance_neigh_cols = [c.strip() for c in distance_df.columns if c != 'School']
if set(neighs) != set(distance_neigh_cols):
    raise ValueError('Neighborhoods in distance.csv columns do not match those in neighborhoods_population.csv.')
distance = {}
for (idx, row) in distance_df.iterrows():
    s = row['School'].strip()
    if s not in schools:
        raise ValueError(f'School {s} in distance.csv not found in school_capacity.csv.')
    for n in neighs:
        try:
            d = float(row[n])
        except Exception:
            raise ValueError(f'Invalid distance for school {s}, neighborhood {n}: {row[n]}')
        distance[s, n] = d
total_white = sum((pop_white[n] for n in neighs))
total_nonwhite = sum((pop_nonwhite[n] for n in neighs))
total_students = total_white + total_nonwhite
if total_students == 0:
    raise ValueError('Total number of students is zero.')
district_white_ratio = total_white / total_students
lower_ratio = 0.5
upper_ratio = 0.7
m = gp.Model('SchoolAssignment')
x_keys = []
for s in schools:
    for n in neighs:
        for g in groups:
            x_keys.append((s, n, g))
x_vars = m.addVars(x_keys, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((distance[s, n] * x_vars[s, n, g] for s in schools for n in neighs for g in groups)), gp.GRB.MINIMIZE)
for n in neighs:
    m.addConstr(gp.quicksum((x_vars[s, n, 'White'] for s in schools)) == pop_white[n], name=f'assign_white_{n}')
    m.addConstr(gp.quicksum((x_vars[s, n, 'NonWhite'] for s in schools)) == pop_nonwhite[n], name=f'assign_nonwhite_{n}')
for s in schools:
    m.addConstr(gp.quicksum((x_vars[s, n, g] for n in neighs for g in groups)) <= school_cap[s], name=f'capacity_{s}')
for s in schools:
    total_white_s = gp.quicksum((x_vars[s, n, 'White'] for n in neighs))
    total_all_s = gp.quicksum((x_vars[s, n, g] for n in neighs for g in groups))
    m.addConstr(total_white_s >= lower_ratio * total_all_s, name=f'racial_lb_{s}')
    m.addConstr(total_white_s <= upper_ratio * total_all_s, name=f'racial_ub_{s}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for key in x_keys:
        var = x_vars[key]
        print(f'{var.VarName} {var.X}')
else:
    print(f'Solver status: {m.Status}')