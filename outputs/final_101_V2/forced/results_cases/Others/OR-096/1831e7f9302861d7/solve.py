import gurobipy as gp
import pandas as pd
import numpy as np
school_cap_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/school_capacity.csv', sep=',')
schools = [str(s).strip() for s in school_cap_df['School']]
school_capacities = {str(row['School']).strip(): int(row['Capacity']) for _, row in school_cap_df.iterrows()}
neigh_pop_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/neighborhoods_population.csv', sep=',')
neighborhoods = [str(n).strip() for n in neigh_pop_df['Neighborhood']]
pop_white = {str(row['Neighborhood']).strip(): int(row['Population_White']) for _, row in neigh_pop_df.iterrows()}
pop_nonwhite = {str(row['Neighborhood']).strip(): int(row['Population_NonWhite']) for _, row in neigh_pop_df.iterrows()}
groups = ['White', 'NonWhite']
dist_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/distance.csv', sep=',')
dist_df['School'] = dist_df['School'].astype(str).str.strip()
distance = {}
for _, row in dist_df.iterrows():
    s = str(row['School']).strip()
    for n in neighborhoods:
        if n not in row:
            raise KeyError(f'Neighborhood {n} not found in distance.csv columns')
        distance[s, n] = float(row[n])
total_white = sum((pop_white[n] for n in neighborhoods))
total_nonwhite = sum((pop_nonwhite[n] for n in neighborhoods))
total_students = total_white + total_nonwhite
district_white_ratio = total_white / total_students if total_students > 0 else 0.0
m = gp.Model('SchoolAssignment')
x = m.addVars(schools, neighborhoods, groups, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((distance[s, n] * x[s, n, g] for s in schools for n in neighborhoods for g in groups)), gp.GRB.MINIMIZE)
for n in neighborhoods:
    m.addConstr(gp.quicksum((x[s, n, 'White'] for s in schools)) == pop_white[n], name='')
    m.addConstr(gp.quicksum((x[s, n, 'NonWhite'] for s in schools)) == pop_nonwhite[n], name='')
for s in schools:
    m.addConstr(gp.quicksum((x[s, n, g] for n in neighborhoods for g in groups)) <= school_capacities[s], name='')
for s in schools:
    total_white_s = gp.quicksum((x[s, n, 'White'] for n in neighborhoods))
    total_all_s = gp.quicksum((x[s, n, g] for n in neighborhoods for g in groups))
    m.addConstr(total_white_s >= 0.5 * total_all_s, name='')
    m.addConstr(total_white_s <= 0.7 * total_all_s, name='')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total travel distance: {m.objVal:.2f} miles')
    print('\nAssignment summary by school:')
    for s in schools:
        total_white_s = sum((x[s, n, 'White'].X for n in neighborhoods))
        total_nonwhite_s = sum((x[s, n, 'NonWhite'].X for n in neighborhoods))
        total_s = total_white_s + total_nonwhite_s
        pct_white = total_white_s / total_s * 100 if total_s > 0 else 0.0
        print(f' School {s}:')
        print(f'   Total assigned: {total_s:.0f} (Capacity: {school_capacities[s]})')
        print(f'   White: {total_white_s:.0f}, NonWhite: {total_nonwhite_s:.0f}, %White: {pct_white:.1f}%')
    print('\nSample assignment details (nonzero only):')
    for s in schools:
        for n in neighborhoods:
            for g in groups:
                val = x[s, n, g].X
                if val > 1e-05:
                    print(f'  {val:.0f} {g} students from {n} assigned to School {s}')
else:
    print(f'No optimal solution found. Status: {m.status}')