CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'In the Superstore chain, multiple branches require inventory replenishment, and several suppliers located '
          'in different cities can provide the necessary goods. Each supplier incurs a fixed cost upon starting '
          'operations, with the fixed cost data provided in the “fixed_cost.csv” file. Each branch needs to source a '
          'certain quantity of goods from these suppliers. For each branch, the transportation cost per unit of goods '
          'from each supplier is recorded in the “transportation_costs.csv” file. Demand information can be gained in '
          "'demand.csv'. The objective is to determine which suppliers to activate so that the demand of all branches "
          'is met while minimizing the total cost. The decision variables y_i are binary, indicating whether a '
          'supplier is operational (open). The decision variables x_{ij} represent the quantity of goods that branch '
          'S_j sources from supplier F_i. For each branch, x_{ij} represents the proportion of the total supply '
          'obtained from different suppliers. These decision variables help determine the optimal allocation of supply '
          'to minimize the total of fixed and transportation costs.',
 'relationships': [{'column_axis': {'id_column': 'customer', 'table_id': 'file_0_view_0'},
                    'matrix_table_id': 'file_2_view_0',
                    'row_axis': {'id_column': 'Unnamed: 0', 'table_id': 'file_1_view_0'},
                    'row_id_column': 'Unnamed: 0',
                    'type': 'matrix'}],
 'route': 'TP',
 'tables': [{'aliases': ['file_0_view_0', 'demand.csv'],
             'columns': ['customer', 'demand'],
             'file_index': 0,
             'file_name': 'demand.csv',
             'filters': {},
             'original_rows': 5,
             'records': [{'source_row': 0, 'values': {'customer': 'C1', 'demand': '143'}},
                         {'source_row': 1, 'values': {'customer': 'C2', 'demand': '6'}},
                         {'source_row': 2, 'values': {'customer': 'C3', 'demand': '10'}},
                         {'source_row': 3, 'values': {'customer': 'C4', 'demand': '25'}},
                         {'source_row': 4, 'values': {'customer': 'C5', 'demand': '3'}}],
             'returned_rows': 5,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'},
            {'aliases': ['file_1_view_0', 'fixed_cost.csv'],
             'columns': ['Unnamed: 0', 'fixed_costs'],
             'file_index': 1,
             'file_name': 'fixed_cost.csv',
             'filters': {},
             'original_rows': 5,
             'records': [{'source_row': 0, 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '97.65000000000001'}},
                         {'source_row': 1, 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '99.76000000000001'}},
                         {'source_row': 2, 'values': {'Unnamed: 0': 'S3', 'fixed_costs': '100.76'}},
                         {'source_row': 3, 'values': {'Unnamed: 0': 'S4', 'fixed_costs': '105.32'}},
                         {'source_row': 4, 'values': {'Unnamed: 0': 'S5', 'fixed_costs': '98.88'}}],
             'returned_rows': 5,
             'role': 'descriptive non-unique role',
             'table_id': 'file_1_view_0'},
            {'aliases': ['file_2_view_0', 'transportation_costs.csv'],
             'columns': ['Unnamed: 0', 'C1', 'C2', 'C3', 'C4', 'C5'],
             'file_index': 2,
             'file_name': 'transportation_costs.csv',
             'filters': {},
             'original_rows': 5,
             'records': [{'source_row': 0,
                          'values': {'C1': '150.74',
                                     'C2': '0.02',
                                     'C3': '49.13',
                                     'C4': '2080.15',
                                     'C5': '426.4',
                                     'Unnamed: 0': 'S1'}},
                         {'source_row': 1,
                          'values': {'C1': '233.05',
                                     'C2': '97.73',
                                     'C3': '49.84',
                                     'C4': '1982.39',
                                     'C5': '23.96',
                                     'Unnamed: 0': 'S2'}},
                         {'source_row': 2,
                          'values': {'C1': '55.68',
                                     'C2': '935.61',
                                     'C3': '4.03',
                                     'C4': '73.09',
                                     'C5': '525.3200000000001',
                                     'Unnamed: 0': 'S3'}},
                         {'source_row': 3,
                          'values': {'C1': '1483.82',
                                     'C2': '1801.08',
                                     'C3': '112.16',
                                     'C4': '816.05',
                                     'C5': '107.01',
                                     'Unnamed: 0': 'S4'}},
                         {'source_row': 4,
                          'values': {'C1': '1119.47',
                                     'C2': '884.3099999999999',
                                     'C3': '0.08',
                                     'C4': '1544.95',
                                     'C5': '543.67',
                                     'Unnamed: 0': 'S5'}}],
             'returned_rows': 5,
             'role': 'descriptive non-unique role',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [{'column_ids_aligned': True,
                                   'expected_shape': [5, 5],
                                   'matrix_table_id': 'file_2_view_0',
                                   'row_ids_aligned': True,
                                   'shape': [5, 5]}],
                'status': 'OK'}}
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
tables = CSVQA_DATA['tables']

def get_table(table_id_or_alias):
    for t in tables:
        if t['table_id'] == table_id_or_alias or table_id_or_alias in t.get('aliases', []):
            return t
    raise KeyError(f'Table {table_id_or_alias} not found')
demand_tbl = get_table('file_0_view_0')
demand_records = [rec['values'] for rec in demand_tbl['records']]
demand_df = pd.DataFrame(demand_records)
demand_df['demand'] = pd.to_numeric(demand_df['demand'])
customers = demand_df['customer'].tolist()
demand_dict = dict(zip(demand_df['customer'], demand_df['demand']))
fixed_tbl = get_table('file_1_view_0')
fixed_records = [rec['values'] for rec in fixed_tbl['records']]
fixed_df = pd.DataFrame(fixed_records)
fixed_df['fixed_costs'] = pd.to_numeric(fixed_df['fixed_costs'])
suppliers = fixed_df['Unnamed: 0'].tolist()
fixed_dict = dict(zip(fixed_df['Unnamed: 0'], fixed_df['fixed_costs']))
trans_tbl = get_table('file_2_view_0')
trans_records = [rec['values'] for rec in trans_tbl['records']]
trans_df = pd.DataFrame(trans_records)
cost_cols = [c for c in trans_df.columns if c != 'Unnamed: 0']
for col in cost_cols:
    trans_df[col] = pd.to_numeric(trans_df[col])
trans_df.set_index('Unnamed: 0', inplace=True)
trans_df = trans_df[customers]
cost_dict = {s: dict(zip(customers, trans_df.loc[s].values)) for s in suppliers}
M = sum(demand_dict.values())
m = gp.Model('Supplier_Selection')
m.Params.MIPGap = 1e-06
y = {}
for s in suppliers:
    y[s] = m.addVar(vtype=GRB.BINARY)
x = {}
for s in suppliers:
    x[s] = {}
    for c in customers:
        x[s][c] = m.addVar(vtype=GRB.CONTINUOUS)
m.update()
obj = gp.quicksum((fixed_dict[s] * y[s] for s in suppliers))
obj += gp.quicksum((cost_dict[s][c] * x[s][c] for s in suppliers for c in customers))
m.setObjective(obj, GRB.MINIMIZE)
for c in customers:
    m.addConstr(gp.quicksum((x[s][c] for s in suppliers)) >= demand_dict[c])
for s in suppliers:
    for c in customers:
        m.addConstr(x[s][c] <= M * y[s])
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName} = {var.X}')
else:
    print(f'Status: {m.Status}')