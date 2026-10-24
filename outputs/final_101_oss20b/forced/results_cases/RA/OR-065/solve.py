CSVQA_DATA = {'query': "On the Bandcamp sales platform, independent musicians and bands require inventory replenishment through warehouses. Multiple distribution warehouses, located in different cities, can provide the necessary inventory. Each warehouse incurs a fixed cost when starting operations, and the fixed cost data is provided in the “fixed_cost.csv” file. Each musician or band needs to source a certain quantity of goods from these warehouses. For each musician or band, the transportation cost per unit of goods from each warehouse is recorded in the “transportation_costs.csv” file. Demand information can be gained in 'demand.csv'. The objective is to determine which warehouses should be activated so that the demand of all musicians and bands is met while minimizing the total cost. The decision variables y_i are binary, indicating whether a warehouse is operational. The decision variables x_{ij} represent the quantity of goods that musician or band S_j sources from warehouse F_i. For each musician or band, x_{ij} represents the proportion of the total supply obtained from different warehouses.", 'relationships': [{'column_axis': {'id_column': 'customer', 'table_id': 'file_0_view_0'}, 'matrix_table_id': 'file_2_view_0', 'row_axis': {'id_column': 'Unnamed: 0', 'table_id': 'file_1_view_0'}, 'row_id_column': 'Unnamed: 0', 'type': 'matrix'}], 'tables': [{'columns': ['customer', 'demand'], 'file_name': 'demand.csv', 'records': [{'data_row_number': 0, 'values': {'customer': 'C1', 'demand': '1083'}}, {'data_row_number': 1, 'values': {'customer': 'C2', 'demand': '776'}}, {'data_row_number': 2, 'values': {'customer': 'C3', 'demand': '16214'}}], 'table_id': 'file_0_view_0'}, {'columns': ['Unnamed: 0', 'fixed_costs'], 'file_name': 'fixed_cost.csv', 'records': [{'data_row_number': 0, 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '102.33'}}, {'data_row_number': 1, 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '94.92'}}, {'data_row_number': 2, 'values': {'Unnamed: 0': 'S3', 'fixed_costs': '91.83'}}], 'table_id': 'file_1_view_0'}, {'columns': ['Unnamed: 0', 'C1', 'C2', 'C3'], 'file_name': 'transportation_costs.csv', 'records': [{'data_row_number': 0, 'values': {'C1': '1506.22', 'C2': '70.90000000000001', 'C3': '8.44', 'Unnamed: 0': 'S1'}}, {'data_row_number': 1, 'values': {'C1': '1732.65', 'C2': '1780.72', 'C3': '567.4400000000001', 'Unnamed: 0': 'S2'}}, {'data_row_number': 2, 'values': {'C1': '115.66', 'C2': '100.76', 'C3': '64.68000000000001', 'Unnamed: 0': 'S3'}}], 'table_id': 'file_2_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']

    def get_table(table_id):
        for t in tables:
            if t['table_id'] == table_id:
                return t
        raise ValueError(f'Table {table_id} not found')
    demand_table = get_table('file_0_view_0')
    demand_records = demand_table['records']
    J = [rec['values']['customer'] for rec in demand_records]
    d_j = {}
    for rec in demand_records:
        cust = rec['values']['customer']
        val = rec['values']['demand']
        try:
            d_j[cust] = float(val)
        except ValueError:
            raise ValueError(f'Invalid demand value for customer {cust}')
    fixed_table = get_table('file_1_view_0')
    fixed_records = fixed_table['records']
    I = [rec['values']['Unnamed: 0'] for rec in fixed_records]
    f_i = {}
    for rec in fixed_records:
        wh = rec['values']['Unnamed: 0']
        val = rec['values']['fixed_costs']
        try:
            f_i[wh] = float(val)
        except ValueError:
            raise ValueError(f'Invalid fixed cost for warehouse {wh}')
    trans_table = get_table('file_2_view_0')
    trans_records = trans_table['records']
    c_ij = {}
    for rec in trans_records:
        wh = rec['values']['Unnamed: 0']
        if wh not in I:
            raise ValueError(f'Warehouse {wh} in transportation matrix not in fixed cost list')
        for cust in J:
            if cust not in rec['values']:
                raise ValueError(f'Customer {cust} missing in transportation matrix row {wh}')
            val = rec['values'][cust]
            try:
                c_ij[wh, cust] = float(val)
            except ValueError:
                raise ValueError(f'Invalid transportation cost for warehouse {wh} to customer {cust}')
    for wh in I:
        if wh not in f_i:
            raise ValueError(f'Missing fixed cost for warehouse {wh}')
    for cust in J:
        if cust not in d_j:
            raise ValueError(f'Missing demand for customer {cust}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    y = model.addVars(I, vtype=GRB.BINARY, name='')
    x = model.addVars([(i, j) for i in I for j in J], vtype=GRB.CONTINUOUS, lb=0, name='')
    obj = gurobipy.quicksum((f_i[i] * y[i] for i in I)) + gurobipy.quicksum((c_ij[i, j] * x[i, j] for i in I for j in J))
    model.setObjective(obj, GRB.MINIMIZE)
    model.addConstrs((gurobipy.quicksum((x[i, j] for i in I)) >= d_j[j] for j in J), name='')
    model.addConstrs((x[i, j] <= d_j[j] * y[i] for i in I for j in J), name='')
    model.optimize()
    return model
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for variable in m.getVars():
        print(f'{variable.VarName}: {variable.X}')
else:
    print(f'Solver status: {m.Status}')
