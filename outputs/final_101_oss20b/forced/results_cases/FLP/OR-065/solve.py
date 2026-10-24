CSVQA_DATA = {'query': "On the Bandcamp sales platform, independent musicians and bands require inventory replenishment through warehouses. Multiple distribution warehouses, located in different cities, can provide the necessary inventory. Each warehouse incurs a fixed cost when starting operations, and the fixed cost data is provided in the “fixed_cost.csv” file. Each musician or band needs to source a certain quantity of goods from these warehouses. For each musician or band, the transportation cost per unit of goods from each warehouse is recorded in the “transportation_costs.csv” file. Demand information can be gained in 'demand.csv'. The objective is to determine which warehouses should be activated so that the demand of all musicians and bands is met while minimizing the total cost. The decision variables y_i are binary, indicating whether a warehouse is operational. The decision variables x_{ij} represent the quantity of goods that musician or band S_j sources from warehouse F_i. For each musician or band, x_{ij} represents the proportion of the total supply obtained from different warehouses.", 'relationships': [{'column_axis': {'id_column': 'customer', 'table_id': 'file_0_view_0'}, 'matrix_table_id': 'file_2_view_0', 'row_axis': {'id_column': 'Unnamed: 0', 'table_id': 'file_1_view_0'}, 'row_id_column': 'Unnamed: 0', 'type': 'matrix'}], 'tables': [{'columns': ['customer', 'demand'], 'file_name': 'demand.csv', 'records': [{'data_row_number': 0, 'values': {'customer': 'C1', 'demand': '1083'}}, {'data_row_number': 1, 'values': {'customer': 'C2', 'demand': '776'}}, {'data_row_number': 2, 'values': {'customer': 'C3', 'demand': '16214'}}], 'table_id': 'file_0_view_0'}, {'columns': ['Unnamed: 0', 'fixed_costs'], 'file_name': 'fixed_cost.csv', 'records': [{'data_row_number': 0, 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '102.33'}}, {'data_row_number': 1, 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '94.92'}}, {'data_row_number': 2, 'values': {'Unnamed: 0': 'S3', 'fixed_costs': '91.83'}}], 'table_id': 'file_1_view_0'}, {'columns': ['Unnamed: 0', 'C1', 'C2', 'C3'], 'file_name': 'transportation_costs.csv', 'records': [{'data_row_number': 0, 'values': {'C1': '1506.22', 'C2': '70.90000000000001', 'C3': '8.44', 'Unnamed: 0': 'S1'}}, {'data_row_number': 1, 'values': {'C1': '1732.65', 'C2': '1780.72', 'C3': '567.4400000000001', 'Unnamed: 0': 'S2'}}, {'data_row_number': 2, 'values': {'C1': '115.66', 'C2': '100.76', 'C3': '64.68000000000001', 'Unnamed: 0': 'S3'}}], 'table_id': 'file_2_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    demand_table = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    fixed_cost_table = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
    cost_table = next((t for t in tables if t['table_id'] == 'file_2_view_0'))
    I = [rec['values']['Unnamed: 0'] for rec in fixed_cost_table['records']]
    J = [rec['values']['customer'] for rec in demand_table['records']]
    demand = {}
    for rec in demand_table['records']:
        cust = rec['values']['customer']
        val = float(rec['values']['demand'])
        demand[cust] = val
    if set(demand.keys()) != set(J):
        missing = set(J) - set(demand.keys())
        raise ValueError(f'Missing demand for customers: {missing}')
    fixed_cost = {}
    for rec in fixed_cost_table['records']:
        wh = rec['values']['Unnamed: 0']
        val = float(rec['values']['fixed_costs'])
        fixed_cost[wh] = val
    if set(fixed_cost.keys()) != set(I):
        missing = set(I) - set(fixed_cost.keys())
        raise ValueError(f'Missing fixed cost for warehouses: {missing}')
    cost = {}
    for rec in cost_table['records']:
        wh = rec['values']['Unnamed: 0']
        if wh not in I:
            raise ValueError(f'Warehouse {wh} in cost matrix not in index set I')
        cost[wh] = {}
        for cust in J:
            if cust not in rec['values']:
                raise ValueError(f'Missing cost for warehouse {wh} to customer {cust}')
            cost[wh][cust] = float(rec['values'][cust])
    for wh in I:
        if wh not in cost:
            raise ValueError(f'Missing cost row for warehouse {wh}')
        if set(cost[wh].keys()) != set(J):
            missing = set(J) - set(cost[wh].keys())
            raise ValueError(f'Missing cost entries for warehouse {wh}: {missing}')
    M = sum(demand.values())
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(i, j) for i in I for j in J], lb=0, vtype=GRB.CONTINUOUS, name='')
    y = m.addVars(I, vtype=GRB.BINARY, name='')
    obj = gurobipy.quicksum((cost[i][j] * x[i, j] for i in I for j in J)) + gurobipy.quicksum((fixed_cost[i] * y[i] for i in I))
    m.setObjective(obj, GRB.MINIMIZE)
    for j in J:
        m.addConstr(gurobipy.quicksum((x[i, j] for i in I)) == demand[j], name=f'demand_{j}')
    for i in I:
        m.addConstr(gurobipy.quicksum((x[i, j] for j in J)) <= M * y[i], name=f'activation_{i}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()
