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
 'relationships': [{'column_axis': {'id_column': 'Market', 'table_id': 'file_1_view_0'},
                    'matrix_table_id': 'file_2_view_0',
                    'row_axis': {'id_column': 'Depot', 'table_id': 'file_0_view_0'},
                    'row_id_column': 'Depot',
                    'type': 'matrix'},
                   {'column_axis': {'id_column': 'Market', 'table_id': 'file_1_view_0'},
                    'matrix_table_id': 'file_3_view_0',
                    'row_axis': {'id_column': 'Depot', 'table_id': 'file_0_view_0'},
                    'row_id_column': 'Depot',
                    'type': 'matrix'}],
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
             'role': 'route variable costs matrix',
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
             'role': 'route fixed costs matrix',
             'table_id': 'file_3_view_0'}],
 'validation': {'matrix_checks': [{'column_ids_aligned': True,
                                   'column_mapping_basis': 'exact',
                                   'expected_shape': [4, 5],
                                   'matrix_table_id': 'file_2_view_0',
                                   'row_ids_aligned': True,
                                   'row_mapping_basis': 'exact',
                                   'shape': [4, 5]},
                                  {'column_ids_aligned': True,
                                   'column_mapping_basis': 'exact',
                                   'expected_shape': [4, 5],
                                   'matrix_table_id': 'file_3_view_0',
                                   'row_ids_aligned': True,
                                   'row_mapping_basis': 'exact',
                                   'shape': [4, 5]}],
                'status': 'OK'}}
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
            raise ValueError(f'Non-numeric or missing SupplyCapacity for depot {depot}')
    market_frame = CSVQA_FRAMES['file_1_view_0']
    markets = []
    D = {}
    for (_, row) in market_frame.iterrows():
        market = row['Market']
        markets.append(market)
        try:
            D[market] = float(row['Demand'])
        except Exception:
            raise ValueError(f'Non-numeric or missing Demand for market {market}')
    c = {}
    varcost_frame = CSVQA_FRAMES['file_2_view_0']
    varcost_depots = set(varcost_frame['Depot'])
    if set(depots) != varcost_depots:
        raise ValueError('Mismatch between depots in depot_capacity.csv and route_variable_costs.csv')
    varcost_markets = [col for col in varcost_frame.columns if col != 'Depot']
    if set(markets) != set(varcost_markets):
        raise ValueError('Mismatch between markets in market_demand.csv and route_variable_costs.csv')
    for (_, row) in varcost_frame.iterrows():
        i = row['Depot']
        for j in markets:
            try:
                c[i, j] = float(row[j])
            except Exception:
                raise ValueError(f'Non-numeric or missing variable cost for route ({i},{j})')
    f = {}
    fixedcost_frame = CSVQA_FRAMES['file_3_view_0']
    fixedcost_depots = set(fixedcost_frame['Depot'])
    if set(depots) != fixedcost_depots:
        raise ValueError('Mismatch between depots in depot_capacity.csv and route_fixed_costs.csv')
    fixedcost_markets = [col for col in fixedcost_frame.columns if col != 'Depot']
    if set(markets) != set(fixedcost_markets):
        raise ValueError('Mismatch between markets in market_demand.csv and route_fixed_costs.csv')
    for (_, row) in fixedcost_frame.iterrows():
        i = row['Depot']
        for j in markets:
            try:
                f[i, j] = float(row[j])
            except Exception:
                raise ValueError(f'Non-numeric or missing fixed cost for route ({i},{j})')
    M = {}
    for i in depots:
        for j in markets:
            M[i, j] = min(S[i], D[j])
    m = gp.Model('FixedChargeTransportation')
    x_vars = m.addVars([(i, j) for i in depots for j in markets], lb=0, vtype=gp.GRB.CONTINUOUS, name='')
    y_vars = m.addVars([(i, j) for i in depots for j in markets], vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((c[i, j] * x_vars[i, j] + f[i, j] * y_vars[i, j] for i in depots for j in markets)), gp.GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x_vars[i, j] for i in depots)) >= D[j] for j in markets), name='')
    m.addConstrs((gp.quicksum((x_vars[i, j] for j in markets)) <= S[i] for i in depots), name='')
    m.addConstrs((x_vars[i, j] <= M[i, j] * y_vars[i, j] for i in depots for j in markets), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)