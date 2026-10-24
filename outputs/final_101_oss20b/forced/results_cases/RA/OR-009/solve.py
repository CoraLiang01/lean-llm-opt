CSVQA_DATA = {'query': '“BrewCo,” a beverage manufacturer, operates multiple production facilities that distribute drinks to various retail locations. The daily demand for each retail outlet is specified in “customer_demand.csv,” while the production capacity of each plant is outlined in “supply_capacity.csv.” The transportation cost per unit of beverages from each plant to each outlet is recorded in “transportation_costs.csv.” The goal is to determine the optimal quantity of beverages to be shipped from each production plant to each retail outlet, ensuring all outlet demands are met without surpassing any plant’s production capacity, while minimizing the total transportation cost.', 'relationships': [{'column_axis': {'id_column': 'customer', 'table_id': 'file_0_view_0'}, 'matrix_table_id': 'file_2_view_0', 'row_axis': {'id_column': 'Unnamed: 0', 'table_id': 'file_1_view_0'}, 'row_id_column': 'Unnamed: 0', 'type': 'matrix'}], 'tables': [{'columns': ['customer', 'demand'], 'file_name': 'customer_demand.csv', 'records': [{'data_row_number': 0, 'values': {'customer': 'C1', 'demand': '94'}}, {'data_row_number': 1, 'values': {'customer': 'C2', 'demand': '39'}}, {'data_row_number': 2, 'values': {'customer': 'C3', 'demand': '65'}}, {'data_row_number': 3, 'values': {'customer': 'C4', 'demand': '435'}}], 'table_id': 'file_0_view_0'}, {'columns': ['Unnamed: 0', 'supply_capacity'], 'file_name': 'supply_capacity.csv', 'records': [{'data_row_number': 0, 'values': {'Unnamed: 0': 'S1', 'supply_capacity': '2531'}}, {'data_row_number': 1, 'values': {'Unnamed: 0': 'S2', 'supply_capacity': '20'}}, {'data_row_number': 2, 'values': {'Unnamed: 0': 'S3', 'supply_capacity': '210'}}, {'data_row_number': 3, 'values': {'Unnamed: 0': 'S4', 'supply_capacity': '241'}}], 'table_id': 'file_1_view_0'}, {'columns': ['Unnamed: 0', 'C1', 'C2', 'C3', 'C4'], 'file_name': 'transportation_costs.csv', 'records': [{'data_row_number': 0, 'values': {'C1': '543.756480860856', 'C2': '23.685276141764653', 'C3': '23.676386730773032', 'C4': '447.75143678673766', 'Unnamed: 0': 'S1'}}, {'data_row_number': 1, 'values': {'C1': '883.9151090405642', 'C2': '0.04977684765576961', 'C3': '0.0350986687216299', 'C4': '44.45588531711622', 'Unnamed: 0': 'S2'}}, {'data_row_number': 2, 'values': {'C1': '537.3456896658107', 'C2': '23.769274659075112', 'C3': '498.95659249465467', 'C4': '440.60737890439776', 'Unnamed: 0': 'S3'}}, {'data_row_number': 3, 'values': {'C1': '1791.493192397229', 'C2': '68.21633865655126', 'C3': '1432.4837339656747', 'C4': '1527.7635425462734', 'Unnamed: 0': 'S4'}}], 'table_id': 'file_2_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    data = CSVQA_DATA
    tables = {t['table_id']: t for t in data['tables']}
    cust_table = tables['file_0_view_0']
    C = [rec['values']['customer'] for rec in cust_table['records']]
    d_c = {}
    for rec in cust_table['records']:
        c = rec['values']['customer']
        d_c[c] = float(rec['values']['demand'])
    plant_table = tables['file_1_view_0']
    P = [rec['values']['Unnamed: 0'] for rec in plant_table['records']]
    s_p = {}
    for rec in plant_table['records']:
        p = rec['values']['Unnamed: 0']
        s_p[p] = float(rec['values']['supply_capacity'])
    cost_table = tables['file_2_view_0']
    cost = {}
    for rec in cost_table['records']:
        p = rec['values']['Unnamed: 0']
        for c in C:
            col = c
            if col not in rec['values']:
                raise ValueError(f'Missing cost for plant {p} to customer {c}')
            cost[p, c] = float(rec['values'][col])
    for p in P:
        if p not in s_p:
            raise ValueError(f'Missing supply capacity for plant {p}')
    for c in C:
        if c not in d_c:
            raise ValueError(f'Missing demand for customer {c}')
    for p in P:
        for c in C:
            if (p, c) not in cost:
                raise ValueError(f'Missing cost for plant {p} to customer {c}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(p, c) for p in P for c in C], lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gurobipy.quicksum((cost[p, c] * x[p, c] for p in P for c in C)), GRB.MINIMIZE)
    m.addConstrs((gurobipy.quicksum((x[p, c] for c in C)) <= s_p[p] for p in P), name='')
    m.addConstrs((gurobipy.quicksum((x[p, c] for p in P)) >= d_c[c] for c in C), name='')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')
