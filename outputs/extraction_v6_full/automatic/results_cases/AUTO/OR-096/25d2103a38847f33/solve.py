import gurobipy as gp
import pandas as pd
import numpy as np
school_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/school_capacity.csv'
neigh_pop_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/neighborhoods_population.csv'
distance_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/distance.csv'
school_df = pd.read_csv(school_capacity_path, sep=',')
schools = [str(s).strip() for s in school_df['School']]
school_cap = {str(row['School']).strip(): int(row['Capacity']) for _, row in school_df.iterrows()}
neigh_df = pd.read_csv(neigh_pop_path, sep=',')
neighs = [str(n).strip() for n in neigh_df['Neighborhood']]
pop_white = {str(row['Neighborhood']).strip(): int(row['Population_White']) for _, row in neigh_df.iterrows()}
pop_nonwhite = {str(row['Neighborhood']).strip(): int(row['Population_NonWhite']) for _, row in neigh_df.iterrows()}
groups = ['White', 'NonWhite']
total_white = sum((pop_white[n] for n in neighs))
total_nonwhite = sum((pop_nonwhite[n] for n in neighs))
total_students = total_white + total_nonwhite
district_white_ratio = total_white / total_students if total_students > 0 else 0.0
dist_df = pd.read_csv(distance_path, sep=',')
dist_df['School'] = dist_df['School'].astype(str).str.strip()
distance = {}
for _, row in dist_df.iterrows():
    s = str(row['School']).strip()
    for n in neighs:
        distance[s, n] = float(row[n])
m = gp.Model('SchoolAssignment_MinTravel')
x = m.addVars(schools, neighs, groups, name='x', lb=0.0)
m.setObjective(gp.quicksum((distance[s, n] * x[s, n, g] for s in schools for n in neighs for g in groups)), gp.GRB.MINIMIZE)
for n in neighs:
    m.addConstr(gp.quicksum((x[s, n, 'White'] for s in schools)) == pop_white[n], name=f'AssignWhite_{n}')
    m.addConstr(gp.quicksum((x[s, n, 'NonWhite'] for s in schools)) == pop_nonwhite[n], name=f'AssignNonWhite_{n}')
for s in schools:
    m.addConstr(gp.quicksum((x[s, n, g] for n in neighs for g in groups)) <= school_cap[s], name=f'Capacity_{s}')
for s in schools:
    total_assigned = gp.quicksum((x[s, n, g] for n in neighs for g in groups))
    total_white_assigned = gp.quicksum((x[s, n, 'White'] for n in neighs))
    m.addConstr(total_white_assigned >= 0.5 * total_assigned, name=f'RacialBalanceLow_{s}')
    m.addConstr(total_white_assigned <= 0.7 * total_assigned, name=f'RacialBalanceHigh_{s}')
m.optimize()