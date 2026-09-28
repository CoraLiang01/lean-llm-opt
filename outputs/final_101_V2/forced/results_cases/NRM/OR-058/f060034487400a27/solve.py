CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'Several suppliers located in different regions can provide the necessary Adidas products to various stores. '
          'Each supplier incurs a fixed cost when starting operations, with the fixed cost data provided in '
          '“fixed_cost.csv.” The company needs to source a unit of each Adidas product for the stores from these '
          'suppliers. For each product, the transportation cost per unit from each supplier to each store is recorded '
          'in “transportation_costs.csv.” Additionally, each store has a specific demand for these products, which is '
          'provided in “demand.csv.” The objective is to determine which suppliers to activate so that the demand for '
          'all Adidas products across all stores is met while minimizing the total cost. The decision variables y_i '
          'are binary, indicating whether a supplier is operational (open). The decision variables x_{ij} represent '
          'the quantity of goods that each store S_j sources from supplier F_i.',
 'relationships': [{'column_axis': {'id_column': 'customer', 'table_id': 'file_0_view_0'},
                    'matrix_table_id': 'file_2_view_0',
                    'row_axis': {'id_column': 'Unnamed: 0', 'table_id': 'file_1_view_0'},
                    'row_id_column': 'Unnamed: 0',
                    'type': 'matrix'}],
 'route': 'NRM',
 'tables': [{'columns': ['customer', 'demand'],
             'file_index': 0,
             'file_name': 'demand.csv',
             'filters': {},
             'original_rows': 6,
             'records': [{'source_row': 0, 'values': {'customer': 'C1', 'demand': '216'}},
                         {'source_row': 1, 'values': {'customer': 'C2', 'demand': '216'}},
                         {'source_row': 2, 'values': {'customer': 'C3', 'demand': '216'}},
                         {'source_row': 3, 'values': {'customer': 'C4', 'demand': '144'}},
                         {'source_row': 4, 'values': {'customer': 'C5', 'demand': '144'}},
                         {'source_row': 5, 'values': {'customer': 'C6', 'demand': '144'}}],
             'returned_rows': 6,
             'role': 'store demand',
             'table_id': 'file_0_view_0'},
            {'columns': ['Unnamed: 0', 'fixed_costs'],
             'file_index': 1,
             'file_name': 'fixed_cost.csv',
             'filters': {},
             'original_rows': 6,
             'records': [{'source_row': 0, 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '98.88'}},
                         {'source_row': 1, 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '99.73'}},
                         {'source_row': 2, 'values': {'Unnamed: 0': 'S3', 'fixed_costs': '94.01000000000001'}},
                         {'source_row': 3, 'values': {'Unnamed: 0': 'S4', 'fixed_costs': '93.77'}},
                         {'source_row': 4, 'values': {'Unnamed: 0': 'S5', 'fixed_costs': '107.59'}},
                         {'source_row': 5, 'values': {'Unnamed: 0': 'S6', 'fixed_costs': '112.65'}}],
             'returned_rows': 6,
             'role': 'supplier fixed costs',
             'table_id': 'file_1_view_0'},
            {'columns': ['Unnamed: 0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'],
             'file_index': 2,
             'file_name': 'transportation_costs.csv',
             'filters': {},
             'original_rows': 6,
             'records': [{'source_row': 0,
                          'values': {'C1': '0.08',
                                     'C2': '52.33',
                                     'C3': '73.56999999999999',
                                     'C4': '1237.33',
                                     'C5': '0.07000000000000001',
                                     'C6': '112.16',
                                     'Unnamed: 0': 'S1'}},
                         {'source_row': 1,
                          'values': {'C1': '46.02',
                                     'C2': '175.23',
                                     'C3': '2026.83',
                                     'C4': '299.89',
                                     'C5': '966.53',
                                     'C6': '1590.42',
                                     'Unnamed: 0': 'S2'}},
                         {'source_row': 2,
                          'values': {'C1': '1031.74',
                                     'C2': '78.13',
                                     'C3': '99.02',
                                     'C4': '277.07',
                                     'C5': '884.45',
                                     'C6': '1800.86',
                                     'Unnamed: 0': 'S3'}},
                         {'source_row': 3,
                          'values': {'C1': '868.75',
                                     'C2': '94.2',
                                     'C3': '1776.34',
                                     'C4': '285.48',
                                     'C5': '868.85',
                                     'C6': '86.55',
                                     'Unnamed: 0': 'S4'}},
                         {'source_row': 4,
                          'values': {'C1': '1577',
                                     'C2': '760.15',
                                     'C3': '2090.19',
                                     'C4': '43.2',
                                     'C5': '1577.12',
                                     'C6': '1095.17',
                                     'Unnamed: 0': 'S5'}},
                         {'source_row': 5,
                          'values': {'C1': '49.14',
                                     'C2': '4.33',
                                     'C3': '2079.57',
                                     'C4': '277.04',
                                     'C5': '1032.01',
                                     'C6': '1543.49',
                                     'Unnamed: 0': 'S6'}}],
             'returned_rows': 6,
             'role': 'supplier-to-store transportation costs',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [{'column_ids_aligned': True,
                                   'expected_shape': [6, 6],
                                   'matrix_table_id': 'file_2_view_0',
                                   'row_ids_aligned': True,
                                   'shape': [6, 6]}],
                'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB
tables = {t['table_id']: t for t in CSVQA_DATA['tables']}
suppliers = []
fixed_cost = {}
for rec in tables['file_1_view_0']['records']:
    s = rec['values']['Unnamed: 0']
    suppliers.append(s)
    try:
        fixed_cost[s] = float(rec['values']['fixed_costs'])
    except Exception:
        raise ValueError(f'Invalid fixed cost for supplier {s}')
stores = []
demand = {}
for rec in tables['file_0_view_0']['records']:
    c = rec['values']['customer']
    stores.append(c)
    try:
        demand[c] = float(rec['values']['demand'])
    except Exception:
        raise ValueError(f'Invalid demand for store {c}')
transportation_cost = {}
trans_table = tables['file_2_view_0']
for rec in trans_table['records']:
    s = rec['values']['Unnamed: 0']
    transportation_cost[s] = {}
    for c in stores:
        try:
            transportation_cost[s][c] = float(rec['values'][c])
        except Exception:
            raise ValueError(f'Missing or invalid transportation cost for supplier {s}, store {c}')
if set(fixed_cost.keys()) != set(suppliers):
    raise ValueError('Mismatch in supplier fixed cost data')
if set(demand.keys()) != set(stores):
    raise ValueError('Mismatch in store demand data')
for s in suppliers:
    if set(transportation_cost[s].keys()) != set(stores):
        raise ValueError(f'Mismatch in transportation cost data for supplier {s}')
M = {}
for s in suppliers:
    M[s] = {}
    for c in stores:
        M[s][c] = demand[c]
m = gp.Model('Adidas_Supplier_Selection')
y = m.addVars(suppliers, vtype=GRB.BINARY, name='')
x = m.addVars(suppliers, stores, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((fixed_cost[s] * y[s] for s in suppliers)) + gp.quicksum((transportation_cost[s][c] * x[s, c] for s in suppliers for c in stores)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x[s, c] for s in suppliers)) == demand[c] for c in stores), name='')
m.addConstrs((x[s, c] <= M[s][c] * y[s] for s in suppliers for c in stores), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')