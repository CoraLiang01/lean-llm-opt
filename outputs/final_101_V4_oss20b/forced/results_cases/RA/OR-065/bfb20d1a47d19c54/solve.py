CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'On the Bandcamp sales platform, independent musicians and bands require inventory replenishment through '
          'warehouses. Multiple distribution warehouses, located in different cities, can provide the necessary '
          'inventory. Each warehouse incurs a fixed cost when starting operations, and the fixed cost data is provided '
          'in the “fixed_cost.csv” file. Each musician or band needs to source a certain quantity of goods from these '
          'warehouses. For each musician or band, the transportation cost per unit of goods from each warehouse is '
          "recorded in the “transportation_costs.csv” file. Demand information can be gained in 'demand.csv'. The "
          'objective is to determine which warehouses should be activated so that the demand of all musicians and '
          'bands is met while minimizing the total cost. The decision variables y_i are binary, indicating whether a '
          'warehouse is operational. The decision variables x_{ij} represent the quantity of goods that musician or '
          'band S_j sources from warehouse F_i. For each musician or band, x_{ij} represents the proportion of the '
          'total supply obtained from different warehouses.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'columns': ['customer', 'demand'],
             'file_index': 0,
             'file_name': 'demand.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 3,
             'records': [{'source_row': 0, 'values': {'customer': 'C1', 'demand': '1083'}},
                         {'source_row': 1, 'values': {'customer': 'C2', 'demand': '776'}},
                         {'source_row': 2, 'values': {'customer': 'C3', 'demand': '16214'}}],
             'returned_rows': 3,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'},
            {'columns': ['Unnamed: 0', 'fixed_costs'],
             'file_index': 1,
             'file_name': 'fixed_cost.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 3,
             'records': [{'source_row': 0, 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '102.33'}},
                         {'source_row': 1, 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '94.92'}},
                         {'source_row': 2, 'values': {'Unnamed: 0': 'S3', 'fixed_costs': '91.83'}}],
             'returned_rows': 3,
             'role': 'descriptive non-unique role',
             'table_id': 'file_1_view_0'},
            {'columns': ['Unnamed: 0', 'C1', 'C2', 'C3'],
             'file_index': 2,
             'file_name': 'transportation_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 3,
             'records': [{'source_row': 0,
                          'values': {'C1': '1506.22', 'C2': '70.90000000000001', 'C3': '8.44', 'Unnamed: 0': 'S1'}},
                         {'source_row': 1,
                          'values': {'C1': '1732.65', 'C2': '1780.72', 'C3': '567.4400000000001', 'Unnamed: 0': 'S2'}},
                         {'source_row': 2,
                          'values': {'C1': '115.66', 'C2': '100.76', 'C3': '64.68000000000001', 'Unnamed: 0': 'S3'}}],
             'returned_rows': 3,
             'role': 'descriptive non-unique role',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    data = CSVQA_DATA
    tables = data['tables']
    fixed_table = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
    demand_table = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    transport_table = next((t for t in tables if t['table_id'] == 'file_2_view_0'))
    fixed_cost = {}
    for rec in fixed_table['records']:
        key = rec['values']['Unnamed: 0'].strip()
        val = float(rec['values']['fixed_costs'].strip())
        fixed_cost[key] = val
    demand = {}
    for rec in demand_table['records']:
        key = rec['values']['customer'].strip()
        val = float(rec['values']['demand'].strip())
        demand[key] = val
    transport = {}
    for rec in transport_table['records']:
        i = rec['values']['Unnamed: 0'].strip()
        for j in demand.keys():
            if j in rec['values']:
                val = float(rec['values'][j].strip())
                transport[i, j] = val
    I = list(fixed_cost.keys())
    J = list(demand.keys())
    for i in I:
        if i not in fixed_cost:
            raise ValueError(f'Missing fixed_cost for warehouse {i}')
    for j in J:
        if j not in demand:
            raise ValueError(f'Missing demand for customer {j}')
    for i in I:
        for j in J:
            if (i, j) not in transport:
                raise ValueError(f'Missing transport cost for warehouse {i} to customer {j}')
    M = {j: demand[j] for j in J}
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    y = model.addVars(I, vtype=GRB.BINARY, name='')
    x = model.addVars([(i, j) for i in I for j in J], vtype=GRB.CONTINUOUS, lb=0, name='')
    obj = gurobipy.quicksum((fixed_cost[i] * y[i] for i in I)) + gurobipy.quicksum((transport[i, j] * x[i, j] for i in I for j in J))
    model.setObjective(obj, GRB.MINIMIZE)
    model.addConstrs((gurobipy.quicksum((x[i, j] for i in I)) >= demand[j] for j in J), name='')
    model.addConstrs((x[i, j] <= M[j] * y[i] for i in I for j in J), name='')
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(f'ObjVal: {model.ObjVal}')
        for var in model.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {model.Status}')
    return model
m = solve_problem()