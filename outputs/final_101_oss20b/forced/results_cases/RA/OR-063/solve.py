CSVQA_DATA = {'query': "On the Bandcamp sales platform, independent musicians and bands require inventory replenishment through warehouses. Multiple distribution warehouses, located in different cities, can provide the necessary inventory. Each warehouse incurs a fixed cost when starting operations, and the fixed cost data is provided in the “fixed_cost.csv” file. Each musician or band needs to source a certain quantity of goods from these warehouses. For each musician or band, the transportation cost per unit of goods from each warehouse is recorded in the “transportation_costs.csv” file. Demand information can be gained in 'demand.csv'. The objective is to determine which warehouses should be activated so that the demand of all musicians and bands is met while minimizing the total cost. The decision variables y_i are binary, indicating whether a warehouse is operational. The decision variables x_{ij} represent the quantity of goods that musician or band S_j sources from warehouse F_i. For each musician or band, x_{ij} represents the proportion of the total supply obtained from different warehouses.", 'relationships': [], 'tables': [{'columns': ['customer', 'demand'], 'file_name': 'demand.csv', 'records': [{'data_row_number': 0, 'values': {'customer': 'C1', 'demand': '1083'}}, {'data_row_number': 1, 'values': {'customer': 'C2', 'demand': '776'}}, {'data_row_number': 2, 'values': {'customer': 'C3', 'demand': '16214'}}, {'data_row_number': 3, 'values': {'customer': 'C4', 'demand': '553'}}, {'data_row_number': 4, 'values': {'customer': 'C5', 'demand': '17106'}}, {'data_row_number': 5, 'values': {'customer': 'C6', 'demand': '594'}}, {'data_row_number': 6, 'values': {'customer': 'C7', 'demand': '732'}}], 'table_id': 'file_0_view_0'}, {'columns': ['Unnamed: 0', 'fixed_costs'], 'file_name': 'fixed_cost.csv', 'records': [{'data_row_number': 0, 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '102.33'}}, {'data_row_number': 1, 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '94.92'}}, {'data_row_number': 2, 'values': {'Unnamed: 0': 'S3', 'fixed_costs': '91.83'}}, {'data_row_number': 3, 'values': {'Unnamed: 0': 'S4', 'fixed_costs': '98.70999999999999'}}, {'data_row_number': 4, 'values': {'Unnamed: 0': 'S5', 'fixed_costs': '95.73'}}, {'data_row_number': 5, 'values': {'Unnamed: 0': 'S6', 'fixed_costs': '99.95999999999999'}}, {'data_row_number': 6, 'values': {'Unnamed: 0': 'S7', 'fixed_costs': '98.16'}}], 'table_id': 'file_1_view_0'}, {'columns': ['Unnamed: 0', 'C1', 'C2', 'C3', 'C4'], 'file_name': 'transportation_costs.csv', 'records': [{'data_row_number': 0, 'values': {'C1': '1506.22', 'C2': '70.90000000000001', 'C3': '8.44', 'C4': '260.27', 'Unnamed: 0': 'S1'}}, {'data_row_number': 1, 'values': {'C1': '1732.65', 'C2': '1780.72', 'C3': '567.4400000000001', 'C4': '448.68', 'Unnamed: 0': 'S2'}}, {'data_row_number': 2, 'values': {'C1': '115.66', 'C2': '100.76', 'C3': '64.68000000000001', 'C4': '1324.53', 'Unnamed: 0': 'S3'}}, {'data_row_number': 3, 'values': {'C1': '1254.78', 'C2': '1115.63', 'C3': '52.31', 'C4': '1036.16', 'Unnamed: 0': 'S4'}}, {'data_row_number': 4, 'values': {'C1': '42.9', 'C2': '891.01', 'C3': '1013.94', 'C4': '1128.72', 'Unnamed: 0': 'S5'}}, {'data_row_number': 5, 'values': {'C1': '0.7', 'C2': '139.46', 'C3': '70.03', 'C4': '79.15000000000001', 'Unnamed: 0': 'S6'}}, {'data_row_number': 6, 'values': {'C1': '1732.3', 'C2': '1780.44', 'C3': '486.5', 'C4': '523.74', 'Unnamed: 0': 'S7'}}], 'table_id': 'file_2_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    data = CSVQA_DATA
    tables = {t['table_id']: t for t in data['tables']}
    demand_table = tables['file_0_view_0']
    demand = {}
    for rec in demand_table['records']:
        cust = rec['values']['customer']
        demand[cust] = int(rec['values']['demand'])
    fixed_table = tables['file_1_view_0']
    fixed_costs = {}
    for rec in fixed_table['records']:
        wh = rec['values']['Unnamed: 0']
        fixed_costs[wh] = float(rec['values']['fixed_costs'])
    trans_table = tables['file_2_view_0']
    transport_costs = {}
    for rec in trans_table['records']:
        wh = rec['values']['Unnamed: 0']
        costs = {}
        for cust in ['C1', 'C2', 'C3', 'C4']:
            if cust in rec['values']:
                costs[cust] = float(rec['values'][cust])
        transport_costs[wh] = costs
    I = list(fixed_costs.keys())
    J = list(demand.keys())
    for i in I:
        if i not in fixed_costs:
            raise ValueError(f'Missing fixed cost for warehouse {i}')
        if i not in transport_costs:
            transport_costs[i] = {}
    for j in J:
        if j not in demand:
            raise ValueError(f'Missing demand for customer {j}')
    M = sum(demand.values())
    m = gurobipy.Model('warehouse_activation')
    m.setParam('MIPGap', 1e-06)
    y = m.addVars(I, vtype=GRB.BINARY, name='')
    x = m.addVars([(i, j) for i in I for j in J], vtype=GRB.INTEGER, lb=0, name='')
    obj = gurobipy.quicksum((fixed_costs[i] * y[i] for i in I)) + gurobipy.quicksum((transport_costs.get(i, {}).get(j, 0.0) * x[i, j] for i in I for j in J))
    m.setObjective(obj, GRB.MINIMIZE)
    for j in J:
        m.addConstr(gurobipy.quicksum((x[i, j] for i in I)) >= demand[j], name=f'demand_{j}')
    for i in I:
        for j in J:
            m.addConstr(x[i, j] <= M * y[i], name=f'max_{i}_{j}')
    m.optimize()
    return m
m = solve_problem()
