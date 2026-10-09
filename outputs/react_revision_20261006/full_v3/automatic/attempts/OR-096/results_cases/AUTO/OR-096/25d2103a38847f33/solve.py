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
if len(set(neighs)) != len(neighs):
    raise ValueError('Neighborhoods are not unique in neighborhoods_population.csv')
pop_white = {str(row['Neighborhood']).strip(): int(row['Population_White']) for (_, row) in neigh_df.iterrows()}
pop_nonwhite = {str(row['Neighborhood']).strip(): int(row['Population_NonWhite']) for (_, row) in neigh_df.iterrows()}
groups = ['White', 'NonWhite']
dist_df = pd.read_csv(distance_path, sep=',')
dist_df['School'] = dist_df['School'].astype(str).str.strip()
dist_df = dist_df.set_index('School')
dist_schools = [str(s).strip() for s in dist_df.index]
if set(schools) != set(dist_schools):
    raise ValueError('Mismatch between schools in school_capacity.csv and distance.csv')
dist_neighs = [str(c).strip() for c in dist_df.columns]
if set(neighs) != set(dist_neighs):
    raise ValueError('Mismatch between neighborhoods in neighborhoods_population.csv and distance.csv')
distance = {}
for s in schools:
    for n in neighs:
        distance[s, n] = float(dist_df.loc[s, n])
total_white = sum((pop_white[n] for n in neighs))
total_nonwhite = sum((pop_nonwhite[n] for n in neighs))
total_students = total_white + total_nonwhite
district_white_ratio = total_white / total_students if total_students > 0 else 0.0
keys = []
for s in schools:
    for n in neighs:
        for g in groups:
            keys.append((s, n, g))
m = gp.Model('SchoolAssignment')
x = m.addVars(keys, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((distance[s, n] * x[s, n, g] for s in schools for n in neighs for g in groups)), gp.GRB.MINIMIZE)
for n in neighs:
    m.addConstr(gp.quicksum((x[s, n, 'White'] for s in schools)) == pop_white[n], name=f'assign_white_{n}')
    m.addConstr(gp.quicksum((x[s, n, 'NonWhite'] for s in schools)) == pop_nonwhite[n], name=f'assign_nonwhite_{n}')
for s in schools:
    m.addConstr(gp.quicksum((x[s, n, g] for n in neighs for g in groups)) <= school_cap[s], name=f'capacity_{s}')
for s in schools:
    W_s = gp.quicksum((x[s, n, 'White'] for n in neighs))
    T_s = gp.quicksum((x[s, n, g] for n in neighs for g in groups))
    m.addConstr(W_s >= 0.5 * T_s, name=f'racial_lb_{s}')
    m.addConstr(W_s <= 0.7 * T_s, name=f'racial_ub_{s}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal {m.objVal}')
    for (s, n, g) in keys:
        v = x[s, n, g]
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')