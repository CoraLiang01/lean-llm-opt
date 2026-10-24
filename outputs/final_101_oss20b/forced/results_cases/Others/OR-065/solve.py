CSVQA_DATA = {'query': "On the Bandcamp sales platform, independent musicians and bands require inventory replenishment through warehouses. Multiple distribution warehouses, located in different cities, can provide the necessary inventory. Each warehouse incurs a fixed cost when starting operations, and the fixed cost data is provided in the “fixed_cost.csv” file. Each musician or band needs to source a certain quantity of goods from these warehouses. For each musician or band, the transportation cost per unit of goods from each warehouse is recorded in the “transportation_costs.csv” file. Demand information can be gained in 'demand.csv'. The objective is to determine which warehouses should be activated so that the demand of all musicians and bands is met while minimizing the total cost. The decision variables y_i are binary, indicating whether a warehouse is operational. The decision variables x_{ij} represent the quantity of goods that musician or band S_j sources from warehouse F_i. For each musician or band, x_{ij} represents the proportion of the total supply obtained from different warehouses.", 'relationships': [{'column_axis': {'id_column': 'customer', 'table_id': 'file_0_view_0'}, 'matrix_table_id': 'file_2_view_0', 'row_axis': {'id_column': 'Unnamed: 0', 'table_id': 'file_1_view_0'}, 'row_id_column': 'Unnamed: 0', 'type': 'matrix'}], 'tables': [{'columns': ['customer', 'demand'], 'file_name': 'demand.csv', 'records': [{'data_row_number': 0, 'values': {'customer': 'C1', 'demand': '1083'}}, {'data_row_number': 1, 'values': {'customer': 'C2', 'demand': '776'}}, {'data_row_number': 2, 'values': {'customer': 'C3', 'demand': '16214'}}], 'table_id': 'file_0_view_0'}, {'columns': ['Unnamed: 0', 'fixed_costs'], 'file_name': 'fixed_cost.csv', 'records': [{'data_row_number': 0, 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '102.33'}}, {'data_row_number': 1, 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '94.92'}}, {'data_row_number': 2, 'values': {'Unnamed: 0': 'S3', 'fixed_costs': '91.83'}}], 'table_id': 'file_1_view_0'}, {'columns': ['Unnamed: 0', 'C1', 'C2', 'C3'], 'file_name': 'transportation_costs.csv', 'records': [{'data_row_number': 0, 'values': {'C1': '1506.22', 'C2': '70.90000000000001', 'C3': '8.44', 'Unnamed: 0': 'S1'}}, {'data_row_number': 1, 'values': {'C1': '1732.65', 'C2': '1780.72', 'C3': '567.4400000000001', 'Unnamed: 0': 'S2'}}, {'data_row_number': 2, 'values': {'C1': '115.66', 'C2': '100.76', 'C3': '64.68000000000001', 'Unnamed: 0': 'S3'}}], 'table_id': 'file_2_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']

    def get_table(table_id):
        for t in tables:
            if t['table_id'] == table_id:
                return t
        raise ValueError(f'Table {table_id} not found in CSVQA_DATA')
    table_warehouses = get_table('file_1_view_0')
    warehouses = []
    fixed_costs = {}
    for rec in table_warehouses['records']:
        wid = rec['values']['Unnamed: 0']
        warehouses.append(wid)
        try:
            fixed_costs[wid] = float(rec['values']['fixed_costs'])
        except (KeyError, ValueError) as e:
            raise ValueError(f'Missing or invalid fixed cost for warehouse {wid}') from e
    table_customers = get_table('file_0_view_0')
    customers = []
    demand = {}
    for rec in table_customers['records']:
        cid = rec['values']['customer']
        customers.append(cid)
        try:
            demand[cid] = float(rec['values']['demand'])
        except (KeyError, ValueError) as e:
            raise ValueError(f'Missing or invalid demand for customer {cid}') from e
    table_transport = get_table('file_2_view_0')
    transportation_costs = {}
    for rec in table_transport['records']:
        wid = rec['values']['Unnamed: 0']
        if wid not in warehouses:
            raise ValueError(f'Warehouse {wid} in transport table not in warehouse list')
        for cid in customers:
            try:
                transportation_costs[wid, cid] = float(rec['values'][cid])
            except (KeyError, ValueError) as e:
                raise ValueError(f'Missing or invalid cost for warehouse {wid} to customer {cid}') from e
    for wid in warehouses:
        if wid not in fixed_costs:
            raise ValueError(f'Fixed cost missing for warehouse {wid}')
    for cid in customers:
        if cid not in demand:
            raise ValueError(f'Demand missing for customer {cid}')
    for wid in warehouses:
        for cid in customers:
            if (wid, cid) not in transportation_costs:
                raise ValueError(f'Transport cost missing for warehouse {wid} to customer {cid}')
    m = gurobipy.Model('WarehouseLocation')
    m.setParam('MIPGap', 1e-06)
    y = m.addVars(warehouses, vtype=GRB.BINARY, name='')
    x = m.addVars([(i, j) for i in warehouses for j in customers], vtype=GRB.CONTINUOUS, lb=0, name='')
    obj = gurobipy.quicksum((fixed_costs[i] * y[i] for i in warehouses)) + gurobipy.quicksum((transportation_costs[i, j] * x[i, j] for i in warehouses for j in customers))
    m.setObjective(obj, GRB.MINIMIZE)
    for j in customers:
        m.addConstr(gurobipy.quicksum((x[i, j] for i in warehouses)) >= demand[j], name=f'demand_{j}')
    for i in warehouses:
        for j in customers:
            m.addConstr(x[i, j] <= demand[j] * y[i], name=f'link_{i}_{j}')
    m.optimize()
    return m
m = solve_problem()
