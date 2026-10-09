import gurobipy as gp
import pandas as pd
import numpy as np
import re
facility_sites_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant14/inputs/facility_sites.csv'
service_zones_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant14/inputs/service_zones.csv'
facility_df = pd.read_csv(facility_sites_path, dtype=str, keep_default_na=False)
zones_df = pd.read_csv(service_zones_path, dtype=str, keep_default_na=False)
facility_df['Center'] = facility_df['Center'].apply(lambda x: x.strip())
facility_df['OpeningCost'] = facility_df['OpeningCost'].astype(int)
facility_df['CoveredDistricts'] = facility_df['CoveredDistricts'].apply(lambda x: [z.strip() for z in x.split(';') if z.strip() != ''])
zones_df['Zone'] = zones_df['Zone'].apply(lambda x: x.strip())
depots = list(facility_df['Center'])
zones = list(zones_df['Zone'])
depot_covers = {row['Center']: set(row['CoveredDistricts']) for (_, row) in facility_df.iterrows()}
zone_covered_by = {z: set() for z in zones}
for (depot, covered_zones) in depot_covers.items():
    for z in covered_zones:
        if z in zone_covered_by:
            zone_covered_by[z].add(depot)
uncovered_zones = [z for z in zones if len(zone_covered_by[z]) == 0]
if uncovered_zones:
    raise ValueError(f'The following zones are not covered by any depot: {uncovered_zones}')
opening_cost = dict(zip(facility_df['Center'], facility_df['OpeningCost']))
m = gp.Model('SetCoveringDepots')
y_vars = m.addVars(depots, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((opening_cost[i] * y_vars[i] for i in depots)), gp.GRB.MINIMIZE)
for z in zones:
    covering_depots = zone_covered_by[z]
    m.addConstr(gp.quicksum((y_vars[i] for i in covering_depots)) >= 1, name=f'cover_{z}')
m.optimize()