import gurobipy as gp
import pandas as pd
import numpy as np
school_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/school_capacity.csv'
neigh_pop_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/neighborhoods_population.csv'
distance_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/distance.csv'
school_df = pd.read_csv(school_capacity_path, sep=',')
neigh_df = pd.read_csv(neigh_pop_path, sep=',')
dist_df = pd.read_csv(distance_path, sep=',')
schools = [str(s).strip() for s in school_df['School']]
neighs = [str(n).strip() for n in neigh_df['Neighborhood']]
groups = ['White', 'NonWhite']
capacity = {str(row['School']).strip(): int(row['Capacity']) for _, row in school_df.iterrows()}
pop_white = {str(row['Neighborhood']).strip(): int(row['Population_White']) for _, row in neigh_df.iterrows()}
pop_nonwhite = {str(row['Neighborhood']).strip(): int(row['Population_NonWhite']) for _, row in neigh_df.iterrows()}
pop = {'White': pop_white, 'NonWhite': pop_nonwhite}
dist = {}
for _, row in dist_df.iterrows():
    s = str(row['School']).strip()
    for n in neighs:
        dist[s, n] = float(row[n])
total_white = sum((pop_white[n] for n in neighs))
total_nonwhite = sum((pop_nonwhite[n] for n in neighs))
total_students = total_white + total_nonwhite
district_white_pct = total_white / total_students if total_students > 0 else 0.0
lower_pct = district_white_pct - 0.1
upper_pct = district_white_pct + 0.1
m = gp.Model('SchoolAssignment')
x = m.addVars(schools, neighs, groups, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((dist[s, n] * x[s, n, g] for s in schools for n in neighs for g in groups)), gp.GRB.MINIMIZE)
for n in neighs:
    for g in groups:
        m.addConstr(gp.quicksum((x[s, n, g] for s in schools)) == pop[g][n], name='')
for s in schools:
    m.addConstr(gp.quicksum((x[s, n, g] for n in neighs for g in groups)) <= capacity[s], name='')
for s in schools:
    total_white_s = gp.quicksum((x[s, n, 'White'] for n in neighs))
    total_students_s = gp.quicksum((x[s, n, g] for n in neighs for g in groups))
    m.addConstr(total_white_s >= lower_pct * total_students_s, name='')
    m.addConstr(total_white_s <= upper_pct * total_students_s, name='')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total travel distance: {m.objVal:.2f} miles')
    print('\n--- Assignment by School ---')
    for s in schools:
        total_white_s = sum((x[s, n, 'White'].X for n in neighs))
        total_nonwhite_s = sum((x[s, n, 'NonWhite'].X for n in neighs))
        total_s = total_white_s + total_nonwhite_s
        pct_white = total_white_s / total_s if total_s > 0 else 0.0
        print(f'\nSchool {s}:')
        print(f'  Total assigned: {total_s:.0f} (Capacity: {capacity[s]})')
        print(f'  White: {total_white_s:.0f}, NonWhite: {total_nonwhite_s:.0f}')
        print(f'  White %: {100 * pct_white:.2f}%')
    print('\n--- Assignment by Neighborhood ---')
    for n in neighs:
        for g in groups:
            assigned = [(s, x[s, n, g].X) for s in schools if x[s, n, g].X > 1e-05]
            if assigned:
                print(f'Neighborhood {n}, {g}:')
                for s, val in assigned:
                    print(f'  Assigned to School {s}: {val:.0f}')
else:
    print(f'No optimal solution found. Status: {m.status}')