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
 'route': 'FLP',
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
             'role': 'descriptive non-unique role',
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
             'role': 'descriptive non-unique role',
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
             'role': 'descriptive non-unique role',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    table_demand = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    table_fixed = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
    table_cost = next((t for t in tables if t['table_id'] == 'file_2_view_0'))
    suppliers = [rec['values']['Unnamed: 0'] for rec in table_fixed['records']]
    stores = [rec['values']['Customer'] for rec in table_demand['records']]
    demand = {}
    for rec in table_demand['records']:
        key = rec['values']['Customer']
        val_str = rec['values']['demand']
        try:
            val = float(val_str)
        except Exception:
            raise ValueError(f'Invalid demand value for {key}')
        demand[key] = val
    fixed_cost = {}
    for rec in table_fixed['records']:
        key = rec['values']['Unnamed: 0']
        val_str = rec['values']['fixed_costs']
        try:
            val = float(val_str)
        except Exception:
            raise ValueError(f'Invalid fixed cost for {key}')
        fixed_cost[key] = val
    cost_columns = [col for col in table_cost['columns'] if col != 'Unnamed: 0']
    if len(cost_columns) != len(stores):
        raise ValueError('Number of cost columns does not match number of stores')
    store_to_col = dict(zip(stores, cost_columns))
    cost = {}
    for rec in table_cost['records']:
        supplier = rec['values']['Unnamed: 0']
        if supplier not in suppliers:
            raise ValueError(f'Supplier {supplier} in cost matrix not in fixed cost table')
        cost[supplier] = {}
        for store in stores:
            col = store_to_col[store]
            val_str = rec['values'][col]
            try:
                val = float(val_str)
            except Exception:
                raise ValueError(f'Invalid cost for supplier {supplier} to store {store}')
            cost[supplier][store] = val
    M = sum(demand.values())
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars([(i, j) for i in suppliers for j in stores], lb=0, vtype=GRB.CONTINUOUS, name='')
    y = model.addVars(suppliers, vtype=GRB.BINARY, name='')
    obj = gurobipy.quicksum((cost[i][j] * x[i, j] for i in suppliers for j in stores)) + gurobipy.quicksum((fixed_cost[i] * y[i] for i in suppliers))
    model.setObjective(obj, GRB.MINIMIZE)
    model.addConstrs((gurobipy.quicksum((x[i, j] for i in suppliers)) == demand[j] for j in stores), name='')
    model.addConstrs((gurobipy.quicksum((x[i, j] for j in stores)) <= M * y[i] for i in suppliers), name='')
    model.optimize()
    return model
m = solve_problem()