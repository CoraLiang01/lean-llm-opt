CSVQA_DATA = {'query': "On the Bandcamp sales platform, independent musicians and bands require inventory replenishment through warehouses. Multiple distribution warehouses, located in different cities, can provide the necessary inventory. Each warehouse incurs a fixed cost when starting operations, and the fixed cost data is provided in the “fixed_cost.csv” file. Each musician or band needs to source a certain quantity of goods from these warehouses. For each musician or band, the transportation cost per unit of goods from each warehouse is recorded in the “transportation_costs.csv” file. Demand information can be gained in 'demand.csv'. The objective is to determine which warehouses should be activated so that the demand of all musicians and bands is met while minimizing the total cost. The decision variables y_i are binary, indicating whether a warehouse is operational. The decision variables x_{ij} represent the quantity of goods that musician or band S_j sources from warehouse F_i. For each musician or band, x_{ij} represents the proportion of the total supply obtained from different warehouses.", 'relationships': [{'column_axis': {'id_column': 'customer', 'table_id': 'file_0_view_0'}, 'matrix_table_id': 'file_2_view_0', 'row_axis': {'id_column': 'Unnamed: 0', 'table_id': 'file_1_view_0'}, 'row_id_column': 'Unnamed: 0', 'type': 'matrix'}], 'tables': [{'columns': ['customer', 'demand'], 'file_name': 'demand.csv', 'records': [{'data_row_number': 0, 'values': {'customer': 'C1', 'demand': '1083'}}, {'data_row_number': 1, 'values': {'customer': 'C2', 'demand': '776'}}, {'data_row_number': 2, 'values': {'customer': 'C3', 'demand': '16214'}}, {'data_row_number': 3, 'values': {'customer': 'C4', 'demand': '553'}}, {'data_row_number': 4, 'values': {'customer': 'C5', 'demand': '17106'}}, {'data_row_number': 5, 'values': {'customer': 'C6', 'demand': '594'}}, {'data_row_number': 6, 'values': {'customer': 'C7', 'demand': '732'}}], 'table_id': 'file_0_view_0'}, {'columns': ['Unnamed: 0', 'fixed_costs'], 'file_name': 'fixed_cost.csv', 'records': [{'data_row_number': 0, 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '102.33'}}, {'data_row_number': 1, 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '94.92'}}, {'data_row_number': 2, 'values': {'Unnamed: 0': 'S3', 'fixed_costs': '91.83'}}, {'data_row_number': 3, 'values': {'Unnamed: 0': 'S4', 'fixed_costs': '98.70999999999999'}}, {'data_row_number': 4, 'values': {'Unnamed: 0': 'S5', 'fixed_costs': '95.73'}}, {'data_row_number': 5, 'values': {'Unnamed: 0': 'S6', 'fixed_costs': '99.95999999999999'}}, {'data_row_number': 6, 'values': {'Unnamed: 0': 'S7', 'fixed_costs': '98.16'}}], 'table_id': 'file_1_view_0'}, {'columns': ['Unnamed: 0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7'], 'file_name': 'transportation_costs.csv', 'records': [{'data_row_number': 0, 'values': {'C1': '1506.22', 'C2': '70.90000000000001', 'C3': '8.44', 'C4': '260.27', 'C5': '197.47', 'C6': '71.70999999999999', 'C7': '61.19', 'Unnamed: 0': 'S1'}}, {'data_row_number': 1, 'values': {'C1': '1732.65', 'C2': '1780.72', 'C3': '567.4400000000001', 'C4': '448.68', 'C5': '29', 'C6': '1484.91', 'C7': '963.92', 'Unnamed: 0': 'S2'}}, {'data_row_number': 2, 'values': {'C1': '115.66', 'C2': '100.76', 'C3': '64.68000000000001', 'C4': '1324.53', 'C5': '64.98999999999999', 'C6': '134.88', 'C7': '2102.83', 'Unnamed: 0': 'S3'}}, {'data_row_number': 3, 'values': {'C1': '1254.78', 'C2': '1115.63', 'C3': '52.31', 'C4': '1036.16', 'C5': '892.63', 'C6': '1464.04', 'C7': '1383.41', 'Unnamed: 0': 'S4'}}, {'data_row_number': 4, 'values': {'C1': '42.9', 'C2': '891.01', 'C3': '1013.94', 'C4': '1128.72', 'C5': '58.91', 'C6': '42.89', 'C7': '1570.31', 'Unnamed: 0': 'S5'}}, {'data_row_number': 5, 'values': {'C1': '0.7', 'C2': '139.46', 'C3': '70.03', 'C4': '79.15000000000001', 'C5': '1482', 'C6': '0.91', 'C7': '110.46', 'Unnamed: 0': 'S6'}}, {'data_row_number': 6, 'values': {'C1': '1732.3', 'C2': '1780.44', 'C3': '486.5', 'C4': '523.74', 'C5': '522.08', 'C6': '82.48', 'C7': '826.41', 'Unnamed: 0': 'S7'}}], 'table_id': 'file_2_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    tables_by_id = {t['table_id']: t for t in tables}
    demand_table = tables_by_id['file_0_view_0']
    demand_dict = {}
    for rec in demand_table['records']:
        vals = rec['values']
        cust = vals['customer']
        try:
            dem = float(vals['demand'])
        except (ValueError, TypeError):
            raise ValueError(f'Invalid demand value for customer {cust}')
        demand_dict[cust] = dem
    fixed_cost_table = tables_by_id['file_1_view_0']
    fixed_cost_dict = {}
    for rec in fixed_cost_table['records']:
        vals = rec['values']
        wh = vals['Unnamed: 0']
        try:
            fc = float(vals['fixed_costs'])
        except (ValueError, TypeError):
            raise ValueError(f'Invalid fixed cost for warehouse {wh}')
        fixed_cost_dict[wh] = fc
    transport_table = tables_by_id['file_2_view_0']
    cost_dict = {}
    for rec in transport_table['records']:
        vals = rec['values']
        wh = vals['Unnamed: 0']
        cost_dict[wh] = {}
        for col in transport_table['columns']:
            if col == 'Unnamed: 0':
                continue
            try:
                c = float(vals[col])
            except (ValueError, TypeError):
                raise ValueError(f'Invalid cost for warehouse {wh}, customer {col}')
            cost_dict[wh][col] = c
    warehouses = list(fixed_cost_dict.keys())
    customers = list(demand_dict.keys())
    if set(warehouses) != set(cost_dict.keys()):
        missing = set(warehouses) - set(cost_dict.keys())
        raise ValueError(f'Missing cost rows for warehouses: {missing}')
    for wh in warehouses:
        if set(cost_dict[wh].keys()) != set(customers):
            missing = set(customers) - set(cost_dict[wh].keys())
            raise ValueError(f'Missing cost columns for warehouse {wh}: {missing}')
    if set(customers) != set(demand_dict.keys()):
        missing = set(customers) - set(demand_dict.keys())
        raise ValueError(f'Missing demand entries for customers: {missing}')
    if len(transport_table['records']) != len(warehouses):
        raise ValueError('Transport matrix row count does not match number of warehouses')
    if len(transport_table['columns']) - 1 != len(customers):
        raise ValueError('Transport matrix column count does not match number of customers')
    total_demand = sum(demand_dict.values())
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    y = m.addVars(warehouses, lb=0, ub=1, vtype=GRB.BINARY, name='')
    x = m.addVars([(i, j) for i in warehouses for j in customers], lb=0, vtype=GRB.CONTINUOUS, name='')
    obj = gurobipy.quicksum((fixed_cost_dict[i] * y[i] for i in warehouses)) + gurobipy.quicksum((cost_dict[i][j] * x[i, j] for i in warehouses for j in customers))
    m.setObjective(obj, GRB.MINIMIZE)
    m.addConstrs((gurobipy.quicksum((x[i, j] for i in warehouses)) >= demand_dict[j] for j in customers), name='')
    m.addConstrs((x[i, j] <= total_demand * y[i] for i in warehouses for j in customers), name='')
    m.optimize()
    return m
m = solve_problem()
