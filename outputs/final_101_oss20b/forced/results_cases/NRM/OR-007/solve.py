CSVQA_DATA = {'query': 'A retail chain, “GreenMart,” operates several warehouses that supply products to its various store locations. The daily demand for each store is provided in “customer_demand.csv,” while the daily supply capacity of each warehouse is detailed in “supply_capacity.csv.” The cost of transporting each unit of product from each warehouse to each store is recorded in “transportation_costs.csv.” The objective is to determine the optimal quantity of products to be shipped from each warehouse to each GreenMart store, ensuring that all store demands are met without exceeding the supply capacity of any warehouse, while minimizing the total transportation cost.', 'relationships': [{'column_axis': {'id_column': 'customer', 'table_id': 'file_0_view_0'}, 'matrix_table_id': 'file_2_view_0', 'row_axis': {'id_column': 'region', 'table_id': 'file_1_view_0'}, 'row_id_column': 'Unnamed: 0', 'type': 'matrix'}], 'tables': [{'columns': ['customer', 'demand'], 'file_name': 'customer_demand.csv', 'records': [{'data_row_number': 0, 'values': {'customer': 'D1', 'demand': '428'}}, {'data_row_number': 1, 'values': {'customer': 'D2', 'demand': '217'}}, {'data_row_number': 2, 'values': {'customer': 'D3', 'demand': '214'}}, {'data_row_number': 3, 'values': {'customer': 'D4', 'demand': '380'}}, {'data_row_number': 4, 'values': {'customer': 'D5', 'demand': '254'}}], 'table_id': 'file_0_view_0'}, {'columns': ['region', 'supply_capacity'], 'file_name': 'supply_capacity.csv', 'records': [{'data_row_number': 0, 'values': {'region': 'S1', 'supply_capacity': '428'}}, {'data_row_number': 1, 'values': {'region': 'S2', 'supply_capacity': '217'}}, {'data_row_number': 2, 'values': {'region': 'S3', 'supply_capacity': '214'}}, {'data_row_number': 3, 'values': {'region': 'S4', 'supply_capacity': '380'}}, {'data_row_number': 4, 'values': {'region': 'S5', 'supply_capacity': '254'}}], 'table_id': 'file_1_view_0'}, {'columns': ['Unnamed: 0', 'D1', 'D2', 'D3', 'D4', 'D5'], 'file_name': 'transportation_costs.csv', 'records': [{'data_row_number': 0, 'values': {'D1': '269.3910588020795', 'D2': '1.4537335390933939', 'D3': '99.60345345756605', 'D4': '26.64078166309837', 'D5': '9.537688956880922', 'Unnamed: 0': 'S1'}}, {'data_row_number': 1, 'values': {'D1': '9.291846876785183', 'D2': '10.874778437070223', 'D3': '144.52609291614627', 'D4': '11.420133077898234', 'D5': '153.1756819927813', 'Unnamed: 0': 'S2'}}, {'data_row_number': 2, 'values': {'D1': '9.674584301671008', 'D2': '2.6191650959687944', 'D3': '100.8242249168735', 'D4': '3.2121910887916876', 'D5': '133.8493396124168', 'Unnamed: 0': 'S3'}}, {'data_row_number': 3, 'values': {'D1': '270.57498480010247', 'D2': '32.50253586', 'D3': '4.6842098096469815', 'D4': '1.5682269686546804', 'D5': '9.58927599', 'Unnamed: 0': 'S4'}}, {'data_row_number': 4, 'values': {'D1': '226.0331910675782', 'D2': '8.669161980826471', 'D3': '65.47681316968448', 'D4': '9.068765258459958', 'D5': '202.65015316425533', 'Unnamed: 0': 'S5'}}], 'table_id': 'file_2_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    demand_table = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    supply_table = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
    cost_table = next((t for t in tables if t['table_id'] == 'file_2_view_0'))
    demand = {}
    for rec in demand_table['records']:
        cust = rec['values']['customer']
        val = int(rec['values']['demand'])
        demand[cust] = val
    supply = {}
    for rec in supply_table['records']:
        region = rec['values']['region']
        val = int(rec['values']['supply_capacity'])
        supply[region] = val
    cost = {}
    row_id_col = 'Unnamed: 0'
    store_cols = [c for c in cost_table['columns'] if c != row_id_col]
    for rec in cost_table['records']:
        warehouse = rec['values'][row_id_col]
        cost[warehouse] = {}
        for s in store_cols:
            cost[warehouse][s] = float(rec['values'][s])
    W = list(supply.keys())
    S = list(demand.keys())
    missing_demand = set(S) - set(demand.keys())
    if missing_demand:
        raise ValueError(f'Missing demand for stores: {missing_demand}')
    missing_supply = set(W) - set(supply.keys())
    if missing_supply:
        raise ValueError(f'Missing supply for warehouses: {missing_supply}')
    for w in W:
        if w not in cost:
            raise ValueError(f'Missing cost row for warehouse {w}')
        missing_cost_cols = set(S) - set(cost[w].keys())
        if missing_cost_cols:
            raise ValueError(f'Missing cost columns for warehouse {w}: {missing_cost_cols}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars([(w, s) for w in W for s in S], vtype=GRB.INTEGER, lb=0, name='')
    model.setObjective(gurobipy.quicksum((cost[w][s] * x[w, s] for w in W for s in S)), GRB.MINIMIZE)
    for s in S:
        model.addConstr(gurobipy.quicksum((x[w, s] for w in W)) == demand[s], name=f'demand_{s}')
    for w in W:
        model.addConstr(gurobipy.quicksum((x[w, s] for s in S)) <= supply[w], name=f'supply_{w}')
    model.optimize()
    return model
m = solve_problem()
