CSVQA_DATA = {'query': "On the Bandcamp sales platform, independent musicians and bands require inventory replenishment through warehouses. Multiple distribution warehouses, located in different cities, can provide the necessary inventory. Each warehouse incurs a fixed cost when starting operations, and the fixed cost data is provided in the “fixed_cost.csv” file. Each musician or band needs to source a certain quantity of goods from these warehouses. For each musician or band, the transportation cost per unit of goods from each warehouse is recorded in the “transportation_costs.csv” file. Demand information can be gained in 'demand.csv'. The objective is to determine which warehouses should be activated so that the demand of all musicians and bands is met while minimizing the total cost. The decision variables y_i are binary, indicating whether a warehouse is operational. The decision variables x_{ij} represent the quantity of goods that musician or band S_j sources from warehouse F_i. For each musician or band, x_{ij} represents the proportion of the total supply obtained from different warehouses.", 'relationships': [], 'tables': [{'columns': ['customer', 'demand'], 'file_name': 'demand.csv', 'records': [{'data_row_number': 0, 'values': {'customer': 'C1', 'demand': '1083'}}, {'data_row_number': 1, 'values': {'customer': 'C2', 'demand': '776'}}, {'data_row_number': 2, 'values': {'customer': 'C3', 'demand': '16214'}}, {'data_row_number': 3, 'values': {'customer': 'C4', 'demand': '553'}}, {'data_row_number': 4, 'values': {'customer': 'C5', 'demand': '17106'}}, {'data_row_number': 5, 'values': {'customer': 'C6', 'demand': '594'}}, {'data_row_number': 6, 'values': {'customer': 'C7', 'demand': '732'}}], 'table_id': 'file_0_view_0'}, {'columns': ['Unnamed: 0', 'fixed_costs'], 'file_name': 'fixed_cost.csv', 'records': [{'data_row_number': 0, 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '102.33'}}, {'data_row_number': 1, 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '94.92'}}, {'data_row_number': 2, 'values': {'Unnamed: 0': 'S3', 'fixed_costs': '91.83'}}, {'data_row_number': 3, 'values': {'Unnamed: 0': 'S4', 'fixed_costs': '98.70999999999999'}}, {'data_row_number': 4, 'values': {'Unnamed: 0': 'S5', 'fixed_costs': '95.73'}}, {'data_row_number': 5, 'values': {'Unnamed: 0': 'S6', 'fixed_costs': '99.95999999999999'}}, {'data_row_number': 6, 'values': {'Unnamed: 0': 'S7', 'fixed_costs': '98.16'}}], 'table_id': 'file_1_view_0'}, {'columns': ['Unnamed: 0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7'], 'file_name': 'transportation_costs.csv', 'records': [{'data_row_number': 0, 'values': {'C1': '1506.22', 'C2': '70.90000000000001', 'C3': '8.44', 'C4': '260.27', 'C5': '197.47', 'C6': '71.70999999999999', 'C7': '61.19', 'Unnamed: 0': 'S1'}}, {'data_row_number': 1, 'values': {'C1': '1732.65', 'C2': '1780.72', 'C3': '567.4400000000001', 'C4': '448.68', 'C5': '29', 'C6': '1484.91', 'C7': '963.92', 'Unnamed: 0': 'S2'}}, {'data_row_number': 2, 'values': {'C1': '115.66', 'C2': '100.76', 'C3': '64.68000000000001', 'C4': '1324.53', 'C5': '64.98999999999999', 'C6': '134.88', 'C7': '2102.83', 'Unnamed: 0': 'S3'}}, {'data_row_number': 3, 'values': {'C1': '1254.78', 'C2': '1115.63', 'C3': '52.31', 'C4': '1036.16', 'C5': '892.63', 'C6': '1464.04', 'C7': '1383.41', 'Unnamed: 0': 'S4'}}, {'data_row_number': 4, 'values': {'C1': '42.9', 'C2': '891.01', 'C3': '1013.94', 'C4': '1128.72', 'C5': '58.91', 'C6': '42.89', 'C7': '1570.31', 'Unnamed: 0': 'S5'}}, {'data_row_number': 5, 'values': {'C1': '0.7', 'C2': '139.46', 'C3': '70.03', 'C4': '79.15000000000001', 'C5': '1482', 'C6': '0.91', 'C7': '110.46', 'Unnamed: 0': 'S6'}}, {'data_row_number': 6, 'values': {'C1': '1732.3', 'C2': '1780.44', 'C3': '486.5', 'C4': '523.74', 'C5': '522.08', 'C6': '82.48', 'C7': '826.41', 'Unnamed: 0': 'S7'}}], 'table_id': 'file_2_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']

    def get_table(table_id):
        for t in tables:
            if t.get('table_id') == table_id:
                return t
        raise ValueError(f'Table {table_id} not found')
    demand_table = get_table('file_0_view_0')
    demand = {}
    customers = set()
    for rec in demand_table['records']:
        vals = rec['values']
        cust = vals['customer']
        if cust in demand:
            raise ValueError(f'Duplicate demand entry for {cust}')
        demand[cust] = float(vals['demand'])
        customers.add(cust)
    fixed_table = get_table('file_1_view_0')
    fixed_cost = {}
    suppliers = set()
    for rec in fixed_table['records']:
        vals = rec['values']
        sup = vals['Unnamed: 0']
        if sup in fixed_cost:
            raise ValueError(f'Duplicate fixed cost entry for {sup}')
        fixed_cost[sup] = float(vals['fixed_costs'])
        suppliers.add(sup)
    trans_table = get_table('file_2_view_0')
    cost = {}
    for rec in trans_table['records']:
        vals = rec['values']
        sup = vals['Unnamed: 0']
        if sup not in suppliers:
            raise ValueError(f'Supplier {sup} in cost matrix not in fixed cost list')
        cost[sup] = {}
        for key, val in vals.items():
            if key == 'Unnamed: 0':
                continue
            if key not in customers:
                raise ValueError(f'Customer {key} in cost matrix not in demand list')
            if key in cost[sup]:
                raise ValueError(f'Duplicate cost entry for supplier {sup}, customer {key}')
            cost[sup][key] = float(val)
    for sup in suppliers:
        if sup not in cost:
            raise ValueError(f'Missing cost entries for supplier {sup}')
        for cust in customers:
            if cust not in cost[sup]:
                raise ValueError(f'Missing cost for supplier {sup}, customer {cust}')
    M = sum(demand.values())
    m = gurobipy.Model('FLP')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(i, j) for i in suppliers for j in customers], lb=0, vtype=GRB.CONTINUOUS, name='')
    y = m.addVars(suppliers, vtype=GRB.BINARY, name='')
    m.addConstrs((gurobipy.quicksum((x[i, j] for i in suppliers)) == demand[j] for j in customers), name='')
    m.addConstrs((gurobipy.quicksum((x[i, j] for j in customers)) <= M * y[i] for i in suppliers), name='')
    obj = gurobipy.quicksum((cost[i][j] * x[i, j] for i in suppliers for j in customers)) + gurobipy.quicksum((fixed_cost[i] * y[i] for i in suppliers))
    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()
    return m
m = solve_problem()
