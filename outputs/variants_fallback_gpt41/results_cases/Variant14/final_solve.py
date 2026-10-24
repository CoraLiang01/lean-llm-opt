import gurobipy as gp
import pandas as pd
import numpy as np
import re
facility_sites_path = 'benchmark_dataset/Variant14/inputs/facility_sites.csv'
facility_sites_df = pd.read_csv(facility_sites_path, sep=',', dtype=str, keep_default_na=False)
service_zones_path = 'benchmark_dataset/Variant14/inputs/service_zones.csv'
service_zones_df = pd.read_csv(service_zones_path, sep=',', dtype=str, keep_default_na=False)
depot_ids = facility_sites_df['Center'].astype(str).tolist()
try:
    opening_costs = {}
    for (idx, row) in facility_sites_df.iterrows():
        center = str(row['Center'])
        try:
            opening_costs[center] = int(row['OpeningCost'])
        except Exception as e:
            raise ValueError(f"Invalid OpeningCost for Center '{center}': {row['OpeningCost']}")
except Exception as e:
    raise
zone_ids = service_zones_df['Zone'].astype(str).tolist()
depot_covers = {}
for (idx, row) in facility_sites_df.iterrows():
    center = str(row['Center'])
    covered_field = row['CoveredDistricts']
    covered_zones = [z.strip() for z in covered_field.split(';') if z.strip() != '']
    depot_covers[center] = set(covered_zones)
zone_covered_by = {zone: set() for zone in zone_ids}
for (depot, covered_zones) in depot_covers.items():
    for zone in covered_zones:
        if zone in zone_covered_by:
            zone_covered_by[zone].add(depot)
for zone in zone_ids:
    if len(zone_covered_by[zone]) == 0:
        raise ValueError(f"Service zone '{zone}' is not covered by any depot.")
m = gp.Model('SetCoveringDepots')
y_vars = m.addVars(depot_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((opening_costs[depot] * y_vars[depot] for depot in depot_ids)), gp.GRB.MINIMIZE)
for zone in zone_ids:
    covering_depots = zone_covered_by[zone]
    m.addConstr(gp.quicksum((y_vars[depot] for depot in covering_depots)) >= 1, name=f'cover_{zone}')
m.optimize()