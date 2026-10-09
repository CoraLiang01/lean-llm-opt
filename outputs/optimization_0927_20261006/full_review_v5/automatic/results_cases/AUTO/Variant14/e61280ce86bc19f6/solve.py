import gurobipy as gp
import pandas as pd
import numpy as np
import re
facility_sites_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant14/inputs/facility_sites.csv'
facility_sites_df = pd.read_csv(facility_sites_path, sep=',', dtype=str, keep_default_na=False)
service_zones_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant14/inputs/service_zones.csv'
service_zones_df = pd.read_csv(service_zones_path, sep=',', dtype=str, keep_default_na=False)
depot_ids = facility_sites_df['Center'].astype(str).str.strip()
if depot_ids.duplicated().any():
    raise ValueError("Duplicate depot IDs found in facility_sites.csv['Center']")
depot_ids = depot_ids.tolist()
try:
    opening_costs = facility_sites_df.set_index(facility_sites_df['Center'].astype(str).str.strip())['OpeningCost'].astype(int).to_dict()
except Exception as e:
    raise ValueError(f'Error extracting OpeningCost: {e}')

def parse_covered_districts(s):
    return set([z.strip() for z in s.split(';') if z.strip()])
covered_zones_by_depot = {}
for (idx, row) in facility_sites_df.iterrows():
    depot = str(row['Center']).strip()
    covered_field = str(row['CoveredDistricts'])
    covered_zones = parse_covered_districts(covered_field)
    if not covered_zones:
        raise ValueError(f'Depot {depot} covers no zones (empty CoveredDistricts field).')
    covered_zones_by_depot[depot] = covered_zones
zone_ids = service_zones_df['Zone'].astype(str).str.strip()
if zone_ids.duplicated().any():
    raise ValueError("Duplicate zone IDs found in service_zones.csv['Zone']")
zone_ids = zone_ids.tolist()
depots_covering_zone = {zone: set() for zone in zone_ids}
for (depot, covered_zones) in covered_zones_by_depot.items():
    for zone in covered_zones:
        if zone in depots_covering_zone:
            depots_covering_zone[zone].add(depot)
for zone in zone_ids:
    if not depots_covering_zone[zone]:
        raise ValueError(f'Service zone {zone} is not covered by any depot.')
m = gp.Model('SetCoveringDepots')
y_vars = m.addVars(depot_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((opening_costs[depot] * y_vars[depot] for depot in depot_ids)), gp.GRB.MINIMIZE)
for zone in zone_ids:
    m.addConstr(gp.quicksum((y_vars[depot] for depot in depots_covering_zone[zone])) >= 1, name=f'cover_{zone}')
m.optimize()