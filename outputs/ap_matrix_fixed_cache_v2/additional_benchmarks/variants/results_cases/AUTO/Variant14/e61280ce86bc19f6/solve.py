import gurobipy as gp
import pandas as pd
import numpy as np
import re
facility_sites_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant14/inputs/facility_sites.csv'
facility_sites_df = pd.read_csv(facility_sites_path, sep=',', dtype=str, keep_default_na=False)
service_zones_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant14/inputs/service_zones.csv'
service_zones_df = pd.read_csv(service_zones_path, sep=',', dtype=str, keep_default_na=False)
depot_ids = facility_sites_df['Center'].apply(lambda x: x.strip()).tolist()
if len(set(depot_ids)) != len(depot_ids):
    raise ValueError('Duplicate depot IDs found in facility_sites.csv.')
try:
    opening_costs = {row['Center'].strip(): int(row['OpeningCost']) for _, row in facility_sites_df.iterrows()}
except Exception as e:
    raise ValueError(f'Error converting OpeningCost to int: {e}')
zone_ids = service_zones_df['Zone'].apply(lambda x: x.strip()).tolist()
if len(set(zone_ids)) != len(zone_ids):
    raise ValueError('Duplicate zone IDs found in service_zones.csv.')

def parse_covered_zones(covered_str):
    return [z.strip() for z in covered_str.split(';') if z.strip()]
depot_covers = {row['Center'].strip(): set(parse_covered_zones(row['CoveredDistricts'])) for _, row in facility_sites_df.iterrows()}
zone_covered_by = {z: set() for z in zone_ids}
for depot, covered_zones in depot_covers.items():
    for z in covered_zones:
        if z in zone_covered_by:
            zone_covered_by[z].add(depot)
for z in zone_ids:
    if not zone_covered_by[z]:
        raise ValueError(f"Service zone '{z}' is not covered by any depot.")
m = gp.Model('SetCoveringDepots')
y_vars = m.addVars(depot_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((opening_costs[i] * y_vars[i] for i in depot_ids)), gp.GRB.MINIMIZE)
for z in zone_ids:
    covering_depots = zone_covered_by[z]
    m.addConstr(gp.quicksum((y_vars[i] for i in covering_depots)) >= 1, name=f'cover_{z}')
m.optimize()