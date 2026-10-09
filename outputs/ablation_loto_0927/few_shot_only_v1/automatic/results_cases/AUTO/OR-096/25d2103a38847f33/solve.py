import gurobipy as gp
import pandas as pd
import numpy as np
import re
school_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/school_capacity.csv'
neigh_pop_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/neighborhoods_population.csv'
distance_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/distance.csv'
school_df = pd.read_csv(school_capacity_path, sep=',')
neigh_df = pd.read_csv(neigh_pop_path, sep=',')
dist_df = pd.read_csv(distance_path, sep=',')
schools = [str(s) for s in school_df['School']]
school_cap = {str(row['School']): float(row['Capacity']) for (_, row) in school_df.iterrows()}
neighs = [str(n) for n in neigh_df['Neighborhood']]
groups = ['White', 'NonWhite']
pop_white = {str(row['Neighborhood']): float(row['Population_White']) for (_, row) in neigh_df.iterrows()}
pop_nonwhite = {str(row['Neighborhood']): float(row['Population_NonWhite']) for (_, row) in neigh_df.iterrows()}
pop_by_group = {'White': pop_white, 'NonWhite': pop_nonwhite}
total_white = sum((pop_white[n] for n in neighs))
total_nonwhite = sum((pop_nonwhite[n] for n in neighs))
total_students = total_white + total_nonwhite
district_white_ratio = total_white / total_students if total_students > 0 else 0.0
dist_df = dist_df.set_index('School')
dist = {}
for s in schools:
    for n in neighs:
        try:
            dist[s, n] = float(dist_df.loc[s, n])
        except KeyError:
            raise KeyError(f"Missing distance for school '{s}' and neighborhood '{n}' in distance.csv")
m = gp.Model('SchoolAssignment')
x = m.addVars(schools, neighs, groups, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((dist[s, n] * x[s, n, g] for s in schools for n in neighs for g in groups)), gp.GRB.MINIMIZE)
for n in neighs:
    for g in groups:
        pop = pop_by_group[g][n]
        m.addConstr(gp.quicksum((x[s, n, g] for s in schools)) == pop, name=f'assign_{n}_{g}')
for s in schools:
    m.addConstr(gp.quicksum((x[s, n, g] for n in neighs for g in groups)) <= school_cap[s], name=f'cap_{s}')
for s in schools:
    total_white_s = gp.quicksum((x[s, n, 'White'] for n in neighs))
    total_all_s = gp.quicksum((x[s, n, g] for n in neighs for g in groups))
    m.addConstr(total_white_s >= 0.5 * total_all_s, name=f'racial_lb_{s}')
    m.addConstr(total_white_s <= 0.7 * total_all_s, name=f'racial_ub_{s}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total travel distance: {m.objVal:.2f} miles')
    print('\n--- Assignment by School ---')
    for s in schools:
        total_white_s = sum((x[s, n, 'White'].X for n in neighs))
        total_nonwhite_s = sum((x[s, n, 'NonWhite'].X for n in neighs))
        total_s = total_white_s + total_nonwhite_s
        white_pct = total_white_s / total_s * 100 if total_s > 0 else 0.0
        print(f'\nSchool: {s}')
        print(f'  Total assigned: {total_s:.0f} (Capacity: {school_cap[s]:.0f})')
        print(f'    White:    {total_white_s:.0f} ({white_pct:.1f}%)')
        print(f'    NonWhite: {total_nonwhite_s:.0f} ({100 - white_pct:.1f}%)')
    print('\n--- Assignment by Neighborhood (nonzero only) ---')
    for n in neighs:
        for g in groups:
            for s in schools:
                val = x[s, n, g].X
                if val > 1e-05:
                    print(f'Neighborhood {n}, Group {g}, School {s}: {val:.0f}')
else:
    print(f'No optimal solution found. Status: {m.status}')