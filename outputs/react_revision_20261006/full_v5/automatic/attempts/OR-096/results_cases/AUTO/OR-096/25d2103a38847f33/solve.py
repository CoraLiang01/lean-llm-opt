import gurobipy as gp
import pandas as pd
import numpy as np
school_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/school_capacity.csv'
neigh_pop_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/neighborhoods_population.csv'
distance_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/distance.csv'
school_df = pd.read_csv(school_capacity_path, sep=',')
schools = [str(s).strip() for s in school_df['School']]
school_cap = {str(row['School']).strip(): int(row['Capacity']) for (_, row) in school_df.iterrows()}
neigh_df = pd.read_csv(neigh_pop_path, sep=',')
neighs = [str(n).strip() for n in neigh_df['Neighborhood']]
pop_white = {str(row['Neighborhood']).strip(): int(row['Population_White']) for (_, row) in neigh_df.iterrows()}
pop_nonwhite = {str(row['Neighborhood']).strip(): int(row['Population_NonWhite']) for (_, row) in neigh_df.iterrows()}
groups = ['White', 'NonWhite']
total_white = sum((pop_white[n] for n in neighs))
total_nonwhite = sum((pop_nonwhite[n] for n in neighs))
total_students = total_white + total_nonwhite
district_white_ratio = total_white / total_students if total_students > 0 else 0.0
dist_df = pd.read_csv(distance_path, sep=',')
dist_df['School'] = dist_df['School'].astype(str).str.strip()
dist_df = dist_df.set_index('School')
if not set(schools).issubset(set(dist_df.index)):
    raise ValueError('Some schools in school_capacity.csv are missing from distance.csv')
if not set(neighs).issubset(set(dist_df.columns)):
    raise ValueError('Some neighborhoods in neighborhoods_population.csv are missing from distance.csv')
distance = {(s, n): float(dist_df.loc[s, n]) for s in schools for n in neighs}
for s in schools:
    for n in neighs:
        if (s, n) not in distance:
            raise ValueError(f'Missing distance for school {s}, neighborhood {n}')
for n in neighs:
    if n not in pop_white or n not in pop_nonwhite:
        raise ValueError(f'Missing population for neighborhood {n}')
m = gp.Model('SchoolAssignment')
x_keys = []
for s in schools:
    for n in neighs:
        for g in groups:
            x_keys.append((s, n, g))
x = m.addVars(x_keys, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((distance[s, n] * x[s, n, g] for s in schools for n in neighs for g in groups)), gp.GRB.MINIMIZE)
for n in neighs:
    m.addConstr(gp.quicksum((x[s, n, 'White'] for s in schools)) == pop_white[n], name=f'assign_white_{n}')
    m.addConstr(gp.quicksum((x[s, n, 'NonWhite'] for s in schools)) == pop_nonwhite[n], name=f'assign_nonwhite_{n}')
for s in schools:
    m.addConstr(gp.quicksum((x[s, n, g] for n in neighs for g in groups)) <= school_cap[s], name=f'capacity_{s}')
for s in schools:
    total_white_s = gp.quicksum((x[s, n, 'White'] for n in neighs))
    total_students_s = gp.quicksum((x[s, n, g] for n in neighs for g in groups))
    m.addConstr(total_white_s >= 0.5 * total_students_s, name=f'racial_lb_{s}')
    m.addConstr(total_white_s <= 0.7 * total_students_s, name=f'racial_ub_{s}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal {m.objVal}')
    for key in x_keys:
        var = x[key]
        print(f'{var.VarName} {var.X}')
else:
    print(f'Solver status: {m.status}')