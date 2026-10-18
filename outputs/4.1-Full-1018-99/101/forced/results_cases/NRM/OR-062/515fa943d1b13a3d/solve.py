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
CSVQA_DATA = globals().get('CSVQA_DATA', None)
if CSVQA_DATA is None:
    raise RuntimeError('CSVQA_DATA not found in global scope.')
demand_table_id = 'file_0_view_0'
fixed_cost_table_id = 'file_1_view_0'
transport_cost_table_id = 'file_2_view_0'
demand_table = None
for t in CSVQA_DATA['tables']:
    if t['table_id'] == demand_table_id:
        demand_table = t
        break
if demand_table is None:
    raise RuntimeError('demand.csv table not found.')
stores = []
demand = {}
for rec in demand_table['records']:
    store = rec['values']['Customer']
    stores.append(store)
    try:
        demand[store] = float(rec['values']['demand'])
    except Exception:
        raise ValueError(f"Invalid demand value for store {store}: {rec['values']['demand']}")
fixed_cost_table = None
for t in CSVQA_DATA['tables']:
    if t['table_id'] == fixed_cost_table_id:
        fixed_cost_table = t
        break
if fixed_cost_table is None:
    raise RuntimeError('fixed_cost.csv table not found.')
suppliers = []
fixed_cost = {}
for rec in fixed_cost_table['records']:
    supplier = rec['values']['Unnamed: 0']
    suppliers.append(supplier)
    try:
        fixed_cost[supplier] = float(rec['values']['fixed_costs'])
    except Exception:
        raise ValueError(f"Invalid fixed cost for supplier {supplier}: {rec['values']['fixed_costs']}")
transport_cost_table = None
for t in CSVQA_DATA['tables']:
    if t['table_id'] == transport_cost_table_id:
        transport_cost_table = t
        break
if transport_cost_table is None:
    raise RuntimeError('transportation_costs.csv table not found.')
transport_store_columns = [col for col in transport_cost_table['columns'] if col != 'Unnamed: 0']
if len(transport_store_columns) != len(stores):
    raise ValueError('Mismatch between number of stores in transportation_costs.csv and demand.csv.')
store_col_to_store = dict(zip(transport_store_columns, stores))
store_to_store_col = {v: k for k, v in store_col_to_store.items()}
transport_cost = {}
for rec in transport_cost_table['records']:
    supplier = rec['values']['Unnamed: 0']
    if supplier not in suppliers:
        raise ValueError(f'Supplier {supplier} in transportation_costs.csv not found in fixed_cost.csv.')
    for store_col in transport_store_columns:
        store = store_col_to_store[store_col]
        try:
            cost = float(rec['values'][store_col])
        except Exception:
            raise ValueError(f"Invalid transportation cost for supplier {supplier}, store {store}: {rec['values'][store_col]}")
        transport_cost[supplier, store] = cost
for i in suppliers:
    if i not in fixed_cost:
        raise ValueError(f'Missing fixed cost for supplier {i}.')
    for j in stores:
        if (i, j) not in transport_cost:
            raise ValueError(f'Missing transportation cost for supplier {i}, store {j}.')
for j in stores:
    if j not in demand:
        raise ValueError(f'Missing demand for store {j}.')
M = {}
for i in suppliers:
    for j in stores:
        M[i, j] = demand[j]
m = gp.Model('Iowa_Liquor_Supplier_Selection')
y_vars = m.addVars(suppliers, vtype=GRB.BINARY, name='')
x_vars = m.addVars(suppliers, stores, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((fixed_cost[i] * y_vars[i] for i in suppliers)) + gp.quicksum((transport_cost[i, j] * x_vars[i, j] for i in suppliers for j in stores)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[i, j] for i in suppliers)) == demand[j] for j in stores), name='')
m.addConstrs((x_vars[i, j] <= M[i, j] * y_vars[i] for i in suppliers for j in stores), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')