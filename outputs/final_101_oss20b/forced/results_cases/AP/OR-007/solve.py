CSVQA_DATA = {'query': 'A retail chain, “GreenMart,” operates several warehouses that supply products to its various store locations. The daily demand for each store is provided in “customer_demand.csv,” while the daily supply capacity of each warehouse is detailed in “supply_capacity.csv.” The cost of transporting each unit of product from each warehouse to each store is recorded in “transportation_costs.csv.” The objective is to determine the optimal quantity of products to be shipped from each warehouse to each GreenMart store, ensuring that all store demands are met without exceeding the supply capacity of any warehouse, while minimizing the total transportation cost.', 'relationships': [{'column_axis': {'id_column': 'customer', 'table_id': 'file_0_view_0'}, 'matrix_table_id': 'file_2_view_0', 'row_axis': {'id_column': 'region', 'table_id': 'file_1_view_0'}, 'row_id_column': 'Unnamed: 0', 'type': 'matrix'}], 'tables': [{'columns': ['customer', 'demand'], 'file_name': 'customer_demand.csv', 'records': [{'data_row_number': 0, 'values': {'customer': 'D1', 'demand': '428'}}, {'data_row_number': 1, 'values': {'customer': 'D2', 'demand': '217'}}, {'data_row_number': 2, 'values': {'customer': 'D3', 'demand': '214'}}, {'data_row_number': 3, 'values': {'customer': 'D4', 'demand': '380'}}, {'data_row_number': 4, 'values': {'customer': 'D5', 'demand': '254'}}], 'table_id': 'file_0_view_0'}, {'columns': ['region', 'supply_capacity'], 'file_name': 'supply_capacity.csv', 'records': [{'data_row_number': 0, 'values': {'region': 'S1', 'supply_capacity': '428'}}, {'data_row_number': 1, 'values': {'region': 'S2', 'supply_capacity': '217'}}, {'data_row_number': 2, 'values': {'region': 'S3', 'supply_capacity': '214'}}, {'data_row_number': 3, 'values': {'region': 'S4', 'supply_capacity': '380'}}, {'data_row_number': 4, 'values': {'region': 'S5', 'supply_capacity': '254'}}], 'table_id': 'file_1_view_0'}, {'columns': ['Unnamed: 0', 'D1', 'D2', 'D3', 'D4', 'D5'], 'file_name': 'transportation_costs.csv', 'records': [{'data_row_number': 0, 'values': {'D1': '269.3910588020795', 'D2': '1.4537335390933939', 'D3': '99.60345345756605', 'D4': '26.64078166309837', 'D5': '9.537688956880922', 'Unnamed: 0': 'S1'}}, {'data_row_number': 1, 'values': {'D1': '9.291846876785183', 'D2': '10.874778437070223', 'D3': '144.52609291614627', 'D4': '11.420133077898234', 'D5': '153.1756819927813', 'Unnamed: 0': 'S2'}}, {'data_row_number': 2, 'values': {'D1': '9.674584301671008', 'D2': '2.6191650959687944', 'D3': '100.8242249168735', 'D4': '3.2121910887916876', 'D5': '133.8493396124168', 'Unnamed: 0': 'S3'}}, {'data_row_number': 3, 'values': {'D1': '270.57498480010247', 'D2': '32.50253586', 'D3': '4.6842098096469815', 'D4': '1.5682269686546804', 'D5': '9.58927599', 'Unnamed: 0': 'S4'}}, {'data_row_number': 4, 'values': {'D1': '226.0331910675782', 'D2': '8.669161980826471', 'D3': '65.47681316968448', 'D4': '9.068765258459958', 'D5': '202.65015316425533', 'Unnamed: 0': 'S5'}}], 'table_id': 'file_2_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    data = CSVQA_DATA
    tables = {t['table_id']: t for t in data['tables']}
    demand_table = tables['file_0_view_0']
    demand = {}
    for rec in demand_table['records']:
        cust = rec['values']['customer'].strip()
        demand[cust] = float(rec['values']['demand'])
    supply_table = tables['file_1_view_0']
    supply = {}
    for rec in supply_table['records']:
        reg = rec['values']['region'].strip()
        supply[reg] = float(rec['values']['supply_capacity'])
    cost_table = tables['file_2_view_0']
    cost = {}
    cols = cost_table['columns']
    row_id_col = cost_table.get('row_id_column', 'Unnamed: 0')
    for rec in cost_table['records']:
        w = rec['values'][row_id_col].strip()
        cost[w] = {}
        for col in cols:
            if col == row_id_col:
                continue
            c = col.strip()
            cost[w][c] = float(rec['values'][col])
    W = set(cost.keys())
    C = set()
    for w in cost:
        C.update(cost[w].keys())
    if set(demand.keys()) != C:
        missing = C - set(demand.keys())
        if missing:
            raise ValueError(f'Missing demand for customers: {missing}')
        missing = set(demand.keys()) - C
        if missing:
            raise ValueError(f'Demand contains unknown customers: {missing}')
    if set(supply.keys()) != W:
        missing = W - set(supply.keys())
        if missing:
            raise ValueError(f'Missing supply for warehouses: {missing}')
        missing = set(supply.keys()) - W
        if missing:
            raise ValueError(f'Supply contains unknown warehouses: {missing}')
    m = gurobipy.Model('GreenMart_Transportation')
    m.setParam('MIPGap', 1e-06)
    var_keys = [(w, c) for w in W for c in C]
    x = m.addVars(var_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gurobipy.quicksum((cost[w][c] * x[w, c] for w in W for c in C)), GRB.MINIMIZE)
    for c in C:
        m.addConstr(gurobipy.quicksum((x[w, c] for w in W)) == demand[c], name=f'demand_{c}')
    for w in W:
        m.addConstr(gurobipy.quicksum((x[w, c] for c in C)) <= supply[w], name=f'supply_{w}')
    m.optimize()
    return m
m = solve_problem()
