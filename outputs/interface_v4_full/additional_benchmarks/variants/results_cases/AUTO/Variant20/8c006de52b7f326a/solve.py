import gurobipy as gp
import pandas as pd
import numpy as np
import re

def solve_problem():
    sensor_sites_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant20/inputs/sensor_sites.csv'
    monitoring_zones_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant20/inputs/monitoring_zones.csv'
    df_sites = pd.read_csv(sensor_sites_path, sep=',')
    df_zones = pd.read_csv(monitoring_zones_path, sep=',')
    site_ids = df_sites['Center'].astype(str).str.strip()
    opening_cost = dict(zip(site_ids, df_sites['OpeningCost']))
    zone_ids = df_zones['Zone'].astype(str).str.strip()
    all_zones = list(zone_ids)
    covered_zones = {}
    for idx, row in df_sites.iterrows():
        site = str(row['Center']).strip()
        covered = [z.strip() for z in str(row['CoveredDistricts']).split(';') if z.strip()]
        covered_zones[site] = set(covered)
    sites_covering_zone = {zone: set() for zone in all_zones}
    for site, zones in covered_zones.items():
        for zone in zones:
            if zone in sites_covering_zone:
                sites_covering_zone[zone].add(site)
    uncovered_zones = [zone for zone, sites in sites_covering_zone.items() if len(sites) == 0]
    if uncovered_zones:
        raise ValueError(f'The following zones are not covered by any site: {uncovered_zones}')
    m = gp.Model('SensorSetCover')
    y = m.addVars(site_ids, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((opening_cost[site] * y[site] for site in site_ids)), gp.GRB.MINIMIZE)
    for zone in all_zones:
        covering_sites = sites_covering_zone[zone]
        m.addConstr(gp.quicksum((y[site] for site in covering_sites)) >= 1, name=f'cover_{zone}')
    m.optimize()
    return m
m = solve_problem()