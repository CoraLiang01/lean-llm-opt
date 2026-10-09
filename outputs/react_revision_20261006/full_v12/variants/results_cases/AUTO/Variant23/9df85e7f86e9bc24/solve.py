CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A regional distributor can ship goods from depots to markets. Depot capacities are listed in '
          'depot_capacity.csv, market demands are listed in market_demand.csv, per-unit route costs are listed in '
          'route_variable_costs.csv, and fixed route activation costs are listed in route_fixed_costs.csv. A route may '
          'carry shipments only if it is activated.\n'
          '\n'
          'Formulate a minimum-cost fixed-charge transportation model. For each depot-market route i-j, define x_ij as '
          'the nonnegative shipment quantity and y_ij as a binary variable equal to 1 if route i-j is activated. The '
          'objective is to minimize variable shipping cost plus fixed route activation cost. The model should include '
          'market demand constraints, depot supply upper-bound constraints, shipment-to-route activation linking '
          'constraints using M_ij = min(depot capacity_i, market demand_j), nonnegativity constraints for shipment '
          'variables, and binary restrictions for route variables.',
 'relationships': [],
 'route': 'Others',
 'tables': [{'columns': ['Depot', 'SupplyCapacity'],
             'file_index': 0,
             'file_name': 'depot_capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 4,
             'records': [{'source_row': 0, 'values': {'Depot': 'D1', 'SupplyCapacity': '120'}},
                         {'source_row': 1, 'values': {'Depot': 'D2', 'SupplyCapacity': '100'}},
                         {'source_row': 2, 'values': {'Depot': 'D3', 'SupplyCapacity': '140'}},
                         {'source_row': 3, 'values': {'Depot': 'D4', 'SupplyCapacity': '90'}}],
             'returned_rows': 4,
             'role': 'depot capacities',
             'table_id': 'file_0_view_0'},
            {'columns': ['Market', 'Demand'],
             'file_index': 1,
             'file_name': 'market_demand.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0, 'values': {'Demand': '65', 'Market': 'M1'}},
                         {'source_row': 1, 'values': {'Demand': '80', 'Market': 'M2'}},
                         {'source_row': 2, 'values': {'Demand': '75', 'Market': 'M3'}},
                         {'source_row': 3, 'values': {'Demand': '95', 'Market': 'M4'}},
                         {'source_row': 4, 'values': {'Demand': '70', 'Market': 'M5'}}],
             'returned_rows': 5,
             'role': 'market demands',
             'table_id': 'file_1_view_0'},
            {'columns': ['Depot', 'M1', 'M2', 'M3', 'M4', 'M5'],
             'file_index': 2,
             'file_name': 'route_variable_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 4,
             'records': [{'source_row': 0,
                          'values': {'Depot': 'D1', 'M1': '4', 'M2': '5', 'M3': '17', 'M4': '18', 'M5': '16'}},
                         {'source_row': 1,
                          'values': {'Depot': 'D2', 'M1': '15', 'M2': '14', 'M3': '3', 'M4': '6', 'M5': '17'}},
                         {'source_row': 2,
                          'values': {'Depot': 'D3', 'M1': '18', 'M2': '16', 'M3': '15', 'M4': '4', 'M5': '5'}},
                         {'source_row': 3,
                          'values': {'Depot': 'D4', 'M1': '6', 'M2': '7', 'M3': '14', 'M4': '16', 'M5': '13'}}],
             'returned_rows': 4,
             'role': 'route variable costs',
             'table_id': 'file_2_view_0'},
            {'columns': ['Depot', 'M1', 'M2', 'M3', 'M4', 'M5'],
             'file_index': 3,
             'file_name': 'route_fixed_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 4,
             'records': [{'source_row': 0,
                          'values': {'Depot': 'D1', 'M1': '240', 'M2': '270', 'M3': '560', 'M4': '590', 'M5': '540'}},
                         {'source_row': 1,
                          'values': {'Depot': 'D2', 'M1': '520', 'M2': '500', 'M3': '230', 'M4': '280', 'M5': '570'}},
                         {'source_row': 2,
                          'values': {'Depot': 'D3', 'M1': '610', 'M2': '580', 'M3': '540', 'M4': '260', 'M5': '250'}},
                         {'source_row': 3,
                          'values': {'Depot': 'D4', 'M1': '290', 'M2': '320', 'M3': '500', 'M4': '550', 'M5': '480'}}],
             'returned_rows': 4,
             'role': 'route fixed costs',
             'table_id': 'file_3_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
import pandas as pd
import numpy as np
import sys

def solve_problem(CSVQA_FRAMES):
    depot_frame = CSVQA_FRAMES['file_0_view_0']
    depots = []
    S = {}
    for (_, row) in depot_frame.iterrows():
        depot = row['Depot']
        depots.append(depot)
        try:
            S[depot] = float(row['SupplyCapacity'])
        except Exception:
            raise ValueError(f"Non-numeric SupplyCapacity for depot {depot}: {row['SupplyCapacity']}")
    market_frame = CSVQA_FRAMES['file_1_view_0']
    markets = []
    D = {}
    for (_, row) in market_frame.iterrows():
        market = row['Market']
        markets.append(market)
        try:
            D[market] = float(row['Demand'])
        except Exception:
            raise ValueError(f"Non-numeric Demand for market {market}: {row['Demand']}")
    c = {}
    varcost_frame = CSVQA_FRAMES['file_2_view_0']
    for (_, row) in varcost_frame.iterrows():
        depot = row['Depot']
        for market in markets:
            try:
                c[depot, market] = float(row[market])
            except Exception:
                raise ValueError(f'Non-numeric variable cost for route ({depot},{market}): {row[market]}')
    f = {}
    fixedcost_frame = CSVQA_FRAMES['file_3_view_0']
    for (_, row) in fixedcost_frame.iterrows():
        depot = row['Depot']
        for market in markets:
            try:
                f[depot, market] = float(row[market])
            except Exception:
                raise ValueError(f'Non-numeric fixed cost for route ({depot},{market}): {row[market]}')
    M = {}
    for i in depots:
        for j in markets:
            M[i, j] = min(S[i], D[j])
    for i in depots:
        for j in markets:
            if (i, j) not in c:
                raise ValueError(f'Missing variable cost for route ({i},{j})')
            if (i, j) not in f:
                raise ValueError(f'Missing fixed cost for route ({i},{j})')
            if (i, j) not in M:
                raise ValueError(f'Missing M_ij for route ({i},{j})')
    m = gp.Model('FixedChargeTransportation')
    route_keys = [(i, j) for i in depots for j in markets]
    x_vars = m.addVars(route_keys, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    y_vars = m.addVars(route_keys, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((c[i, j] * x_vars[i, j] + f[i, j] * y_vars[i, j] for i in depots for j in markets)), gp.GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x_vars[i, j] for i in depots)) >= D[j] for j in markets), name='')
    m.addConstrs((gp.quicksum((x_vars[i, j] for j in markets)) <= S[i] for i in depots), name='')
    m.addConstrs((x_vars[i, j] <= M[i, j] * y_vars[i, j] for i in depots for j in markets), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print(f'ObjVal {m.objVal}')
        for (i, j) in route_keys:
            print(f'x[{i},{j}] {x_vars[i, j].VarName} {x_vars[i, j].X}')
            print(f'y[{i},{j}] {y_vars[i, j].VarName} {y_vars[i, j].X}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_problem(CSVQA_FRAMES)