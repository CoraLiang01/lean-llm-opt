CSVQA_DATA = {'query': 'There are 10 source locations (S1,S2,...,S10) and 20 demand locations (D1-D20). Each truck is loaded with maximum 10 units of cargo and can only be dispatched in whole trucks (integer trips).The supply information for each supplier (S1-S10) is detailed in expanded_sources.csv. The demand information for each demand locations (D1-D20) is listed in expanded_destinations.csv. The unit costs for each truck in different routes are known in expanded_cost_matrix.csv. Truck loading is allowed to be partial; the number of dispatched trucks per route must be integer. Costs are per unit (cargo), not per truck. The source of goods is constrained and the demand is fixed. Our goal is to meet all demands and minimize the total transportation cost, which is measured in units of each cargo.', 'relationships': [{'column_axis': {'id_column': 'destination_id', 'table_id': 'file_1_view_0'}, 'matrix_table_id': 'file_0_view_0', 'row_axis': {'id_column': 'source_id', 'table_id': 'file_2_view_0'}, 'row_id_column': 'source_id', 'type': 'matrix'}], 'tables': [{'columns': ['source_id', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19', 'D20'], 'file_name': 'expanded_cost_matrix.csv', 'records': [{'data_row_number': 0, 'values': {'D1': '2.75', 'D10': '6.44', 'D11': '5.2', 'D12': '4.54', 'D13': '5.39', 'D14': '4.34', 'D15': '8.28', 'D16': '8.87', 'D17': '9.03', 'D18': '8.49', 'D19': '9.66', 'D2': '2.6', 'D20': '10.91', 'D3': '2.9', 'D4': '1.7', 'D5': '1.85', 'D6': '1.98', 'D7': '2.74', 'D8': '6.2', 'D9': '5.75', 'source_id': 'S1'}}, {'data_row_number': 1, 'values': {'D1': '5.24', 'D10': '4.24', 'D11': '3.13', 'D12': '3.16', 'D13': '3.67', 'D14': '2.43', 'D15': '5.83', 'D16': '6.51', 'D17': '6.68', 'D18': '6.1', 'D19': '7.2', 'D2': '4.87', 'D20': '8.4', 'D3': '4.72', 'D4': '4.12', 'D5': '4.26', 'D6': '4.55', 'D7': '4.45', 'D8': '3.88', 'D9': '3.24', 'source_id': 'S2'}}, {'data_row_number': 2, 'values': {'D1': '5.54', 'D10': '4.42', 'D11': '3.37', 'D12': '3.53', 'D13': '3.98', 'D14': '2.7', 'D15': '5.84', 'D16': '6.48', 'D17': '6.68', 'D18': '6.14', 'D19': '7.24', 'D2': '5.16', 'D20': '8.31', 'D3': '5.09', 'D4': '4.37', 'D5': '4.57', 'D6': '4.78', 'D7': '4.89', 'D8': '3.84', 'D9': '3.08', 'source_id': 'S3'}}, {'data_row_number': 3, 'values': {'D1': '4.51', 'D10': '4.71', 'D11': '3.55', 'D12': '3.15', 'D13': '3.84', 'D14': '2.6', 'D15': '6.51', 'D16': '7.03', 'D17': '7.27', 'D18': '6.69', 'D19': '7.89', 'D2': '4.17', 'D20': '9.04', 'D3': '4.01', 'D4': '3.36', 'D5': '3.59', 'D6': '3.73', 'D7': '3.68', 'D8': '4.39', 'D9': '3.93', 'source_id': 'S4'}}, {'data_row_number': 4, 'values': {'D1': '4.84', 'D10': '5.35', 'D11': '4.25', 'D12': '4.14', 'D13': '4.73', 'D14': '3.46', 'D15': '6.89', 'D16': '7.5', 'D17': '7.67', 'D18': '7.2', 'D19': '8.2', 'D2': '4.74', 'D20': '9.35', 'D3': '4.77', 'D4': '3.8', 'D5': '3.96', 'D6': '4.1', 'D7': '4.52', 'D8': '4.82', 'D9': '4.03', 'source_id': 'S5'}}, {'data_row_number': 5, 'values': {'D1': '9.13', 'D10': '2.19', 'D11': '3.34', 'D12': '4.07', 'D13': '3.3', 'D14': '4.24', 'D15': '2.77', 'D16': '2.63', 'D17': '2.97', 'D18': '2.56', 'D19': '3.32', 'D2': '8.32', 'D20': '5.09', 'D3': '7.78', 'D4': '7.9', 'D5': '8.13', 'D6': '8.45', 'D7': '7.47', 'D8': '3.04', 'D9': '4.25', 'source_id': 'S6'}}, {'data_row_number': 6, 'values': {'D1': '8.64', 'D10': '1.43', 'D11': '2.64', 'D12': '3.64', 'D13': '2.92', 'D14': '3.51', 'D15': '2.41', 'D16': '2.74', 'D17': '2.98', 'D18': '2.46', 'D19': '3.51', 'D2': '7.86', 'D20': '5.08', 'D3': '7.34', 'D4': '7.32', 'D5': '7.51', 'D6': '7.83', 'D7': '7.03', 'D8': '1.9', 'D9': '3.14', 'source_id': 'S7'}}, {'data_row_number': 7, 'values': {'D1': '9.25', 'D10': '2.88', 'D11': '3.78', 'D12': '4.27', 'D13': '3.49', 'D14': '4.63', 'D15': '3.62', 'D16': '3.34', 'D17': '3.59', 'D18': '3.29', 'D19': '3.79', 'D2': '8.41', 'D20': '5.54', 'D3': '7.81', 'D4': '8.13', 'D5': '8.32', 'D6': '8.6', 'D7': '7.49', 'D8': '3.71', 'D9': '5.0', 'source_id': 'S8'}}, {'data_row_number': 8, 'values': {'D1': '10.3', 'D10': '3.49', 'D11': '4.59', 'D12': '5.28', 'D13': '4.47', 'D14': '5.51', 'D15': '3.22', 'D16': '2.59', 'D17': '2.72', 'D18': '2.68', 'D19': '2.82', 'D2': '9.58', 'D20': '4.47', 'D3': '8.89', 'D4': '9.11', 'D5': '9.32', 'D6': '9.69', 'D7': '8.57', 'D8': '4.11', 'D9': '5.29', 'source_id': 'S9'}}, {'data_row_number': 9, 'values': {'D1': '7.85', 'D10': '1.52', 'D11': '2.18', 'D12': '2.78', 'D13': '2.01', 'D14': '3.0', 'D15': '3.47', 'D16': '3.62', 'D17': '3.89', 'D18': '3.41', 'D19': '4.38', 'D2': '7.08', 'D20': '5.98', 'D3': '6.57', 'D4': '6.72', 'D5': '6.88', 'D6': '7.18', 'D7': '6.21', 'D8': '2.34', 'D9': '3.56', 'source_id': 'S10'}}], 'table_id': 'file_0_view_0'}, {'columns': ['destination_id', 'demand_units'], 'file_name': 'expanded_destinations.csv', 'records': [{'data_row_number': 0, 'values': {'demand_units': '61', 'destination_id': 'D1'}}, {'data_row_number': 1, 'values': {'demand_units': '54', 'destination_id': 'D2'}}, {'data_row_number': 2, 'values': {'demand_units': '56', 'destination_id': 'D3'}}, {'data_row_number': 3, 'values': {'demand_units': '54', 'destination_id': 'D4'}}, {'data_row_number': 4, 'values': {'demand_units': '53', 'destination_id': 'D5'}}, {'data_row_number': 5, 'values': {'demand_units': '47', 'destination_id': 'D6'}}, {'data_row_number': 6, 'values': {'demand_units': '56', 'destination_id': 'D7'}}, {'data_row_number': 7, 'values': {'demand_units': '57', 'destination_id': 'D8'}}, {'data_row_number': 8, 'values': {'demand_units': '56', 'destination_id': 'D9'}}, {'data_row_number': 9, 'values': {'demand_units': '34', 'destination_id': 'D10'}}, {'data_row_number': 10, 'values': {'demand_units': '55', 'destination_id': 'D11'}}, {'data_row_number': 11, 'values': {'demand_units': '53', 'destination_id': 'D12'}}, {'data_row_number': 12, 'values': {'demand_units': '37', 'destination_id': 'D13'}}, {'data_row_number': 13, 'values': {'demand_units': '31', 'destination_id': 'D14'}}, {'data_row_number': 14, 'values': {'demand_units': '62', 'destination_id': 'D15'}}, {'data_row_number': 15, 'values': {'demand_units': '58', 'destination_id': 'D16'}}, {'data_row_number': 16, 'values': {'demand_units': '39', 'destination_id': 'D17'}}, {'data_row_number': 17, 'values': {'demand_units': '32', 'destination_id': 'D18'}}, {'data_row_number': 18, 'values': {'demand_units': '38', 'destination_id': 'D19'}}, {'data_row_number': 19, 'values': {'demand_units': '67', 'destination_id': 'D20'}}], 'table_id': 'file_1_view_0'}, {'columns': ['source_id', 'supply_units'], 'file_name': 'expanded_sources.csv', 'records': [{'data_row_number': 0, 'values': {'source_id': 'S1', 'supply_units': '103'}}, {'data_row_number': 1, 'values': {'source_id': 'S2', 'supply_units': '87'}}, {'data_row_number': 2, 'values': {'source_id': 'S3', 'supply_units': '95'}}, {'data_row_number': 3, 'values': {'source_id': 'S4', 'supply_units': '112'}}, {'data_row_number': 4, 'values': {'source_id': 'S5', 'supply_units': '97'}}, {'data_row_number': 5, 'values': {'source_id': 'S6', 'supply_units': '103'}}, {'data_row_number': 6, 'values': {'source_id': 'S7', 'supply_units': '101'}}, {'data_row_number': 7, 'values': {'source_id': 'S8', 'supply_units': '94'}}, {'data_row_number': 8, 'values': {'source_id': 'S9', 'supply_units': '102'}}, {'data_row_number': 9, 'values': {'source_id': 'S10', 'supply_units': '106'}}], 'table_id': 'file_2_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = {t['table_id']: t for t in CSVQA_DATA['tables']}
    supply_table = tables.get('file_2_view_0')
    if supply_table is None:
        raise ValueError("Supply table 'file_2_view_0' not found.")
    supply = {}
    for rec in supply_table['records']:
        vals = rec['values']
        sid = vals['source_id']
        try:
            supply_units = int(vals['supply_units'])
        except (KeyError, ValueError):
            raise ValueError(f'Invalid supply_units for source {sid}')
        supply[sid] = supply_units
    demand_table = tables.get('file_1_view_0')
    if demand_table is None:
        raise ValueError("Demand table 'file_1_view_0' not found.")
    demand = {}
    for rec in demand_table['records']:
        vals = rec['values']
        did = vals['destination_id']
        try:
            demand_units = int(vals['demand_units'])
        except (KeyError, ValueError):
            raise ValueError(f'Invalid demand_units for destination {did}')
        demand[did] = demand_units
    cost_table = tables.get('file_0_view_0')
    if cost_table is None:
        raise ValueError("Cost matrix table 'file_0_view_0' not found.")
    cost = {}
    dest_cols = [c for c in cost_table['columns'] if c != 'source_id']
    for rec in cost_table['records']:
        vals = rec['values']
        sid = vals['source_id']
        if sid not in supply:
            raise ValueError(f'Source {sid} in cost matrix not in supply data.')
        for did in dest_cols:
            if did not in demand:
                raise ValueError(f'Destination {did} in cost matrix not in demand data.')
            try:
                c_val = float(vals[did])
            except (KeyError, ValueError):
                raise ValueError(f'Invalid cost value for {sid}-{did}')
            cost[sid, did] = c_val
    for s in supply:
        for d in demand:
            if (s, d) not in cost:
                raise ValueError(f'Missing cost for source {s} and destination {d}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    S = list(supply.keys())
    D = list(demand.keys())
    x_keys = [(s, d) for s in S for d in D]
    y_keys = [(s, d) for s in S for d in D]
    x = m.addVars(x_keys, vtype=GRB.INTEGER, lb=0, name='')
    y = m.addVars(y_keys, vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(gurobipy.quicksum((cost[s, d] * x[s, d] for s in S for d in D)), GRB.MINIMIZE)
    m.addConstrs((gurobipy.quicksum((x[s, d] for d in D)) <= supply[s] for s in S), name='')
    m.addConstrs((gurobipy.quicksum((x[s, d] for s in S)) == demand[d] for d in D), name='')
    m.addConstrs((x[s, d] <= 10 * y[s, d] for s in S for d in D), name='')
    m.optimize()
    return m
m = solve_problem()
