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
 'route': 'FLP',
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
    tables = CSVQA_DATA['tables']
    demand_table = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    fixed_cost_table = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
    transport_table = next((t for t in tables if t['table_id'] == 'file_2_view_0'))
    I = set()
    for rec in fixed_cost_table['records']:
        I.add(rec['values']['Unnamed: 0'])
    J = set()
    for rec in demand_table['records']:
        J.add(rec['values']['customer'])
    f = {}
    for rec in fixed_cost_table['records']:
        key = rec['values']['Unnamed: 0']
        try:
            f[key] = float(rec['values']['fixed_costs'])
        except ValueError:
            raise ValueError(f'Invalid fixed cost for warehouse {key}')
    d = {}
    for rec in demand_table['records']:
        key = rec['values']['customer']
        try:
            d[key] = float(rec['values']['demand'])
        except ValueError:
            raise ValueError(f'Invalid demand for customer {key}')
    c = {}
    for rec in transport_table['records']:
        i = rec['values']['Unnamed: 0']
        for j in J:
            if j not in rec['values']:
                raise ValueError(f'Missing transportation cost for warehouse {i} to customer {j}')
            try:
                c[i, j] = float(rec['values'][j])
            except ValueError:
                raise ValueError(f'Invalid transportation cost for warehouse {i} to customer {j}')
    for i in I:
        if i not in f:
            raise ValueError(f'Missing fixed cost for warehouse {i}')
    for j in J:
        if j not in d:
            raise ValueError(f'Missing demand for customer {j}')
    for i in I:
        for j in J:
            if (i, j) not in c:
                raise ValueError(f'Missing transportation cost for warehouse {i} to customer {j}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    y = model.addVars(I, vtype=GRB.BINARY, name='')
    x = model.addVars([(i, j) for i in I for j in J], vtype=GRB.CONTINUOUS, lb=0, name='')
    obj = gurobipy.quicksum((f[i] * y[i] for i in I)) + gurobipy.quicksum((c[i, j] * x[i, j] for i in I for j in J))
    model.setObjective(obj, GRB.MINIMIZE)
    for j in J:
        model.addConstr(gurobipy.quicksum((x[i, j] for i in I)) >= d[j])
    for i in I:
        for j in J:
            model.addConstr(x[i, j] <= d[j] * y[i])
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(model.ObjVal)
        for var in model.getVars():
            print(var.VarName, var.X)
    else:
        print(model.Status)
    return model
m = solve_problem()