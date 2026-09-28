CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The Iowa Department of Commerce requires that any store selling alcohol in bottled form for off-premises '
          'consumption must hold a Class “E” liquor license, a typical arrangement for most state liquor regulatory '
          'authorities. All alcohol sales from stores registered with the Iowa Department of Commerce are recorded in '
          'the department’s system, which is publicly released as open data by the State of Iowa. Several suppliers '
          'located in different cities can provide the necessary liquor products to these licensed stores. Each '
          'supplier incurs a fixed cost when starting operations, with the fixed cost data provided in '
          '“fixed_cost.csv.” The Department needs to source a unit of each liquor product for the stores from these '
          'suppliers. For each product, the transportation cost per unit from each supplier to each store is recorded '
          'in “transportation_costs.csv.” Additionally, each store has a specific demand for these products, which is '
          'provided in “demand.csv.” The objective is to determine which suppliers to activate so that the demand for '
          'all liquor products across all licensed stores is met while minimizing the total cost. The decision '
          'variables y_i are binary, indicating whether a supplier is operational (open). The decision variables '
          'x_{ij} represent the quantity of goods that each store S_j sources from supplier F_i.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Customer', 'demand'],
             'file_index': 0,
             'file_name': 'demand.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0, 'values': {'Customer': 'Customer_1', 'demand': '2397'}},
                         {'source_row': 1, 'values': {'Customer': 'Customer_2', 'demand': '1889'}},
                         {'source_row': 2, 'values': {'Customer': 'Customer_3', 'demand': '2518'}},
                         {'source_row': 3, 'values': {'Customer': 'Customer_4', 'demand': '3218'}},
                         {'source_row': 4, 'values': {'Customer': 'Customer_5', 'demand': '1813'}}],
             'returned_rows': 5,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['Unnamed: 0', 'fixed_costs'],
             'file_index': 1,
             'file_name': 'fixed_cost.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0, 'values': {'Unnamed: 0': 'MOUNT AYR', 'fixed_costs': '96.58'}},
                         {'source_row': 1, 'values': {'Unnamed: 0': 'WAUKEE', 'fixed_costs': '94.06'}},
                         {'source_row': 2, 'values': {'Unnamed: 0': 'WAVERLY', 'fixed_costs': '94.37'}},
                         {'source_row': 3, 'values': {'Unnamed: 0': 'PELLA', 'fixed_costs': '82.88'}},
                         {'source_row': 4, 'values': {'Unnamed: 0': 'DES MOINES', 'fixed_costs': '94.95999999999999'}}],
             'returned_rows': 5,
             'role': 'file_1',
             'table_id': 'file_1_view_0'},
            {'columns': ['Unnamed: 0', 'CLARINDA', 'FORT MADISON', 'SIOUX CITY', 'TOLEDO', 'BANCROFT'],
             'file_index': 2,
             'file_name': 'transportation_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0,
                          'values': {'BANCROFT': '1685.53',
                                     'CLARINDA': '694.6799999999999',
                                     'FORT MADISON': '17.48',
                                     'SIOUX CITY': '20.07',
                                     'TOLEDO': '199.02',
                                     'Unnamed: 0': 'MOUNT AYR'}},
                         {'source_row': 1,
                          'values': {'BANCROFT': '90.69',
                                     'CLARINDA': '15.13',
                                     'FORT MADISON': '1.5',
                                     'SIOUX CITY': '1.43',
                                     'TOLEDO': '27.88',
                                     'Unnamed: 0': 'WAUKEE'}},
                         {'source_row': 2,
                          'values': {'BANCROFT': '78.73',
                                     'CLARINDA': '2.34',
                                     'FORT MADISON': '349.34',
                                     'SIOUX CITY': '246.6',
                                     'TOLEDO': '41.3',
                                     'Unnamed: 0': 'WAVERLY'}},
                         {'source_row': 3,
                          'values': {'BANCROFT': '38.93',
                                     'CLARINDA': '1181.6',
                                     'FORT MADISON': '1458.53',
                                     'SIOUX CITY': '1646.36',
                                     'TOLEDO': '1924.55',
                                     'Unnamed: 0': 'PELLA'}},
                         {'source_row': 4,
                          'values': {'BANCROFT': '103.84',
                                     'CLARINDA': '1030.8',
                                     'FORT MADISON': '43.48',
                                     'SIOUX CITY': '932.4299999999999',
                                     'TOLEDO': '55.39',
                                     'Unnamed: 0': 'DES MOINES'}}],
             'returned_rows': 5,
             'role': 'file_2',
             'table_id': 'file_2_view_0'}],
 'validation': {'fallback_reason': "Plan omitted explicit query subset filters: ['E']",
                'planner_errors': ["Plan omitted explicit query subset filters: ['E']"],
                'status': 'FALLBACK_FULL_DATA'}}
import gurobipy as gp
from gurobipy import GRB
data = CSVQA_DATA
fixed_cost_table = next((t for t in data['tables'] if t['table_id'] == 'file_1_view_0'))
F = [rec['values']['Unnamed: 0'] for rec in fixed_cost_table['records']]
fixed_costs = {}
for rec in fixed_cost_table['records']:
    supplier = rec['values']['Unnamed: 0']
    try:
        fixed_costs[supplier] = float(rec['values']['fixed_costs'])
    except Exception:
        raise ValueError(f'Invalid fixed cost for supplier {supplier}')
demand_table = next((t for t in data['tables'] if t['table_id'] == 'file_0_view_0'))
S = [rec['values']['Customer'] for rec in demand_table['records']]
demands = {}
for rec in demand_table['records']:
    store = rec['values']['Customer']
    try:
        demands[store] = float(rec['values']['demand'])
    except Exception:
        raise ValueError(f'Invalid demand for store {store}')
trans_cost_table = next((t for t in data['tables'] if t['table_id'] == 'file_2_view_0'))
store_col_map = {}
for col in trans_cost_table['columns']:
    if col != 'Unnamed: 0':
        for s in S:
            if s.upper() in col.upper() or col.upper() in s.upper():
                store_col_map[s] = col
if len(store_col_map) < len(S):
    cost_cols = [c for c in trans_cost_table['columns'] if c != 'Unnamed: 0']
    if len(cost_cols) != len(S):
        raise ValueError('Mismatch between number of stores and transportation cost columns')
    for idx, s in enumerate(S):
        store_col_map[s] = cost_cols[idx]
trans_costs = {}
for rec in trans_cost_table['records']:
    supplier = rec['values']['Unnamed: 0']
    trans_costs[supplier] = {}
    for s in S:
        col = store_col_map[s]
        try:
            trans_costs[supplier][s] = float(rec['values'][col])
        except Exception:
            raise ValueError(f'Invalid transportation cost for supplier {supplier}, store {s}')
if set(F) != set(trans_costs.keys()):
    raise ValueError('Mismatch in suppliers between fixed cost and transportation cost tables')
for i in F:
    if set(S) != set(trans_costs[i].keys()):
        raise ValueError(f'Mismatch in stores for supplier {i} in transportation cost table')
if set(S) != set(demands.keys()):
    raise ValueError('Mismatch in stores between demand and transportation cost tables')
M = {}
for i in F:
    M[i] = {}
    for j in S:
        M[i][j] = demands[j]
m = gp.Model('Iowa_Liquor_Supplier_Selection')
y = m.addVars(F, vtype=GRB.BINARY, name='')
x = m.addVars(F, S, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((fixed_costs[i] * y[i] for i in F)) + gp.quicksum((trans_costs[i][j] * x[i, j] for i in F for j in S)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x[i, j] for i in F)) == demands[j] for j in S), name='')
m.addConstrs((x[i, j] <= M[i][j] * y[i] for i in F for j in S), name='')
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')