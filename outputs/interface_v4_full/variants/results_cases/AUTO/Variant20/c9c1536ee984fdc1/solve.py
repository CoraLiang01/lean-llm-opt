import gurobipy as gp
import pandas as pd
import numpy as np
import re
sensor_sites_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant20/inputs/sensor_sites.csv'
zones_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant20/inputs/monitoring_zones.csv'
sensor_sites_df = pd.read_csv(sensor_sites_path, sep=',')
sensor_sites_df['Center'] = sensor_sites_df['Center'].astype(str).str.strip()

def parse_covered_zones(s):
    return set([z.strip() for z in str(s).split(';') if z.strip() != ''])
sensor_sites_df['CoveredZonesSet'] = sensor_sites_df['CoveredDistricts'].apply(parse_covered_zones)
zones_df = pd.read_csv(zones_path, sep=',')
zones_df['Zone'] = zones_df['Zone'].astype(str).str.strip()
sensor_sites = list(sensor_sites_df['Center'])
zones = list(zones_df['Zone'])
opening_cost = dict(zip(sensor_sites_df['Center'], sensor_sites_df['OpeningCost']))
site_covers = dict(zip(sensor_sites_df['Center'], sensor_sites_df['CoveredZonesSet']))
zone_covered_by = {z: set() for z in zones}
for site in sensor_sites:
    for z in site_covers[site]:
        if z in zone_covered_by:
            zone_covered_by[z].add(site)
for z in zones:
    if len(zone_covered_by[z]) == 0:
        raise ValueError(f"Monitoring zone '{z}' is not covered by any sensor site.")
m = gp.Model('MinimumSetCovering')
y = m.addVars(sensor_sites, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((opening_cost[site] * y[site] for site in sensor_sites)), gp.GRB.MINIMIZE)
for z in zones:
    covering_sites = zone_covered_by[z]
    m.addConstr(gp.quicksum((y[site] for site in covering_sites)) >= 1, name=f'cover_{z}')
m.optimize()