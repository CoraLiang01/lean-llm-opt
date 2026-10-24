CSVQA_DATA = {'query': "On the Bandcamp sales platform, independent musicians and bands require inventory replenishment through warehouses. Multiple distribution warehouses, located in different cities, can provide the necessary inventory. Each warehouse incurs a fixed cost when starting operations, and the fixed cost data is provided in the “fixed_cost.csv” file. Each musician or band needs to source a certain quantity of goods from these warehouses. For each musician or band, the transportation cost per unit of goods from each warehouse is recorded in the “transportation_costs.csv” file. Demand information can be gained in 'demand.csv'. The objective is to determine which warehouses should be activated so that the demand of all musicians and bands is met while minimizing the total cost. The decision variables y_i are binary, indicating whether a warehouse is operational. The decision variables x_{ij} represent the quantity of goods that musician or band S_j sources from warehouse F_i. For each musician or band, x_{ij} represents the proportion of the total supply obtained from different warehouses.", 'relationships': [], 'tables': [{'columns': ['customer', 'demand'], 'file_name': 'demand.csv', 'records': [{'data_row_number': 0, 'values': {'customer': 'C1', 'demand': '1083'}}, {'data_row_number': 1, 'values': {'customer': 'C2', 'demand': '776'}}, {'data_row_number': 2, 'values': {'customer': 'C3', 'demand': '16214'}}], 'table_id': 'file_0_view_0'}, {'columns': ['Unnamed: 0', 'fixed_costs'], 'file_name': 'fixed_cost.csv', 'records': [{'data_row_number': 0, 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '102.33'}}, {'data_row_number': 1, 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '94.92'}}, {'data_row_number': 2, 'values': {'Unnamed: 0': 'S3', 'fixed_costs': '91.83'}}], 'table_id': 'file_1_view_0'}, {'columns': ['Unnamed: 0', 'C1', 'C2', 'C3'], 'file_name': 'transportation_costs.csv', 'records': [{'data_row_number': 0, 'values': {'C1': '1506.22', 'C2': '70.90000000000001', 'C3': '8.44', 'Unnamed: 0': 'S1'}}, {'data_row_number': 1, 'values': {'C1': '1732.65', 'C2': '1780.72', 'C3': '567.4400000000001', 'Unnamed: 0': 'S2'}}, {'data_row_number': 2, 'values': {'C1': '115.66', 'C2': '100.76', 'C3': '64.68000000000001', 'Unnamed: 0': 'S3'}}], 'table_id': 'file_2_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']

    def get_table(table_id):
        for t in tables:
            if t['table_id'] == table_id:
                return t
        raise ValueError(f'Table {table_id} not found in CSVQA_DATA')
    table_fixed = get_table('file_1_view_0')
    I = [rec['values']['Unnamed: 0'] for rec in table_fixed['records']]
    table_demand = get_table('file_0_view_0')
    J = [rec['values']['customer'] for rec in table_demand['records']]
    d = {}
    for rec in table_demand['records']:
        j = rec['values']['customer']
        if j not in J:
            raise ValueError(f'Unexpected customer {j} in demand table')
        d[j] = float(rec['values']['demand'])
    if set(d.keys()) != set(J):
        missing = set(J) - set(d.keys())
        raise ValueError(f'Missing demand for customers: {missing}')
    f = {}
    for rec in table_fixed['records']:
        i = rec['values']['Unnamed: 0']
        if i not in I:
            raise ValueError(f'Unexpected warehouse {i} in fixed cost table')
        f[i] = float(rec['values']['fixed_costs'])
    if set(f.keys()) != set(I):
        missing = set(I) - set(f.keys())
        raise ValueError(f'Missing fixed cost for warehouses: {missing}')
    table_cost = get_table('file_2_view_0')
    c = {}
    for rec in table_cost['records']:
        i = rec['values']['Unnamed: 0']
        if i not in I:
            raise ValueError(f'Unexpected warehouse {i} in cost table')
        c[i] = {}
        for j in J:
            if j not in rec['values']:
                raise ValueError(f'Missing cost for warehouse {i} to customer {j}')
            c[i][j] = float(rec['values'][j])
    if set(c.keys()) != set(I):
        missing = set(I) - set(c.keys())
        raise ValueError(f'Missing cost rows for warehouses: {missing}')
    for i in I:
        if set(c[i].keys()) != set(J):
            missing = set(J) - set(c[i].keys())
            raise ValueError(f'Missing cost columns for warehouse {i}: {missing}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    y = m.addVars(I, vtype=GRB.BINARY, name='')
    x = m.addVars([(i, j) for i in I for j in J], vtype=GRB.CONTINUOUS, lb=0, name='')
    obj = gurobipy.quicksum((f[i] * y[i] for i in I)) + gurobipy.quicksum((c[i][j] * x[i, j] for i in I for j in J))
    m.setObjective(obj, GRB.MINIMIZE)
    for j in J:
        m.addConstr(gurobipy.quicksum((x[i, j] for i in I)) >= d[j])
    for i in I:
        for j in J:
            m.addConstr(x[i, j] <= y[i] * d[j])
    m.optimize()
    return m
m = solve_problem()
