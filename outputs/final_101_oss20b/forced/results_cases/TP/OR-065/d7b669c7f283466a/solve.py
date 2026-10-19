CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'On the Bandcamp sales platform, independent musicians and bands require inventory replenishment through '
          'warehouses. Multiple distribution warehouses, located in different cities, can provide the necessary '
          'inventory. Each warehouse incurs a fixed cost when starting operations, and the fixed cost data is provided '
          'in the “fixed_cost.csv” file. Each musician or band needs to source a certain quantity of goods from these '
          'warehouses. For each musician or band, the transportation cost per unit of goods from each warehouse is '
          "recorded in the “transportation_costs.csv” file. Demand information can be gained in 'demand.csv'. The "
          'objective is to determine which warehouses should be activated so that the demand of all musicians and '
          'bands is met while minimizing the total cost. The decision variables y_i are binary, indicating whether a '
          'warehouse is operational. The decision variables x_{ij} represent the quantity of goods that musician or '
          'band S_j sources from warehouse F_i. For each musician or band, x_{ij} represents the proportion of the '
          'total supply obtained from different warehouses.',
 'relationships': [],
 'route': 'TP',
 'tables': [{'aliases': ['file_0_view_0', 'demand.csv'],
             'columns': ['customer', 'demand'],
             'file_index': 0,
             'file_name': 'demand.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 3,
             'records': [{'source_row': 0, 'values': {'customer': 'C1', 'demand': '1083'}},
                         {'source_row': 1, 'values': {'customer': 'C2', 'demand': '776'}},
                         {'source_row': 2, 'values': {'customer': 'C3', 'demand': '16214'}}],
             'returned_rows': 3,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'},
            {'aliases': ['file_1_view_0', 'fixed_cost.csv'],
             'columns': ['Unnamed: 0', 'fixed_costs'],
             'file_index': 1,
             'file_name': 'fixed_cost.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 3,
             'records': [{'source_row': 0, 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '102.33'}},
                         {'source_row': 1, 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '94.92'}},
                         {'source_row': 2, 'values': {'Unnamed: 0': 'S3', 'fixed_costs': '91.83'}}],
             'returned_rows': 3,
             'role': 'descriptive non-unique role',
             'table_id': 'file_1_view_0'},
            {'aliases': ['file_2_view_0', 'transportation_costs.csv'],
             'columns': ['Unnamed: 0', 'C1', 'C2', 'C3'],
             'file_index': 2,
             'file_name': 'transportation_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 3,
             'records': [{'source_row': 0,
                          'values': {'C1': '1506.22', 'C2': '70.90000000000001', 'C3': '8.44', 'Unnamed: 0': 'S1'}},
                         {'source_row': 1,
                          'values': {'C1': '1732.65', 'C2': '1780.72', 'C3': '567.4400000000001', 'Unnamed: 0': 'S2'}},
                         {'source_row': 2,
                          'values': {'C1': '115.66', 'C2': '100.76', 'C3': '64.68000000000001', 'Unnamed: 0': 'S3'}}],
             'returned_rows': 3,
             'role': 'descriptive non-unique role',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
demand_table = next((t for t in CSVQA_DATA['tables'] if t['table_id'] == 'file_0_view_0'))
fixed_table = next((t for t in CSVQA_DATA['tables'] if t['table_id'] == 'file_1_view_0'))
cost_table = next((t for t in CSVQA_DATA['tables'] if t['table_id'] == 'file_2_view_0'))
demand_df = pd.DataFrame.from_records([rec['values'] for rec in demand_table['records']], columns=demand_table['columns'])
fixed_df = pd.DataFrame.from_records([rec['values'] for rec in fixed_table['records']], columns=fixed_table['columns'])
cost_df = pd.DataFrame.from_records([rec['values'] for rec in cost_table['records']], columns=cost_table['columns'])
demand_df['demand'] = demand_df['demand'].astype(float)
fixed_df['fixed_costs'] = fixed_df['fixed_costs'].astype(float)
for col in cost_df.columns:
    if col != 'Unnamed: 0':
        cost_df[col] = cost_df[col].astype(float)
warehouses = list(fixed_df['Unnamed: 0'])
customers = list(demand_df['customer'])
demand = dict(zip(demand_df['customer'], demand_df['demand']))
fixed_costs = dict(zip(fixed_df['Unnamed: 0'], fixed_df['fixed_costs']))
cost = {row['Unnamed: 0']: {c: row[c] for c in customers} for _, row in cost_df.iterrows()}
m = gp.Model('warehouse_location')
m.setParam(GRB.Param.MIPGap, 1e-06)
y = {i: m.addVar(vtype=GRB.BINARY) for i in warehouses}
x = {(i, j): m.addVar(vtype=GRB.CONTINUOUS) for i in warehouses for j in customers}
for j in customers:
    m.addConstr(gp.quicksum((x[i, j] for i in warehouses)) == demand[j])
for i in warehouses:
    for j in customers:
        m.addConstr(x[i, j] <= demand[j] * y[i])
obj = gp.quicksum((fixed_costs[i] * y[i] for i in warehouses)) + gp.quicksum((cost[i][j] * x[i, j] for i in warehouses for j in customers))
m.setObjective(obj, GRB.MINIMIZE)
m.optimize()
if m.status == GRB.OPTIMAL:
    print('ObjVal:', m.objVal)
    for v in m.getVars():
        print(v.VarName, v.X)
else:
    print('Status:', m.status)