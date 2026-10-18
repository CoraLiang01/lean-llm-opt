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
 'route': 'AP',
 'tables': [{'aliases': ['file_0_view_0', 'demand.csv'],
             'columns': ['customer', 'demand'],
             'file_index': 0,
             'file_name': 'demand.csv',
             'filters': {'conditions': [{'column': 'customer',
                                         'dtype': 'string',
                                         'evidence': 'branch',
                                         'operator': 'contains',
                                         'value': ''}],
                         'logic': 'and'},
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
             'filters': {'conditions': [{'column': 'Unnamed: 0',
                                         'dtype': 'string',
                                         'evidence': 'fixed cost',
                                         'operator': 'contains',
                                         'value': ''}],
                         'logic': 'and'},
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
             'filters': {'conditions': [{'column': 'Unnamed: 0',
                                         'dtype': 'string',
                                         'evidence': 'transportation cost',
                                         'operator': 'contains',
                                         'value': ''}],
                         'logic': 'and'},
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
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
tables = CSVQA_DATA['tables']
demand_table = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
demand_df = pd.DataFrame.from_records([r['values'] for r in demand_table['records']])
demand_df['demand'] = demand_df['demand'].astype(float)
customers = demand_df['customer'].tolist()
demand = dict(zip(customers, demand_df['demand']))
fixed_table = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
fixed_df = pd.DataFrame.from_records([r['values'] for r in fixed_table['records']])
fixed_df['fixed_costs'] = fixed_df['fixed_costs'].astype(float)
suppliers = fixed_df['Unnamed: 0'].tolist()
fixed_cost = dict(zip(suppliers, fixed_df['fixed_costs']))
trans_table = next((t for t in tables if t['table_id'] == 'file_2_view_0'))
trans_df = pd.DataFrame.from_records([r['values'] for r in trans_table['records']])
for col in ['C1', 'C2', 'C3', 'C4', 'C5']:
    trans_df[col] = trans_df[col].astype(float)
cost = {}
for _, row in trans_df.iterrows():
    s = row['Unnamed: 0']
    cost[s] = {c: row[c] for c in customers}
m = gp.Model('Supplier_Activation')
m.Params.MIPGap = 1e-06
x = m.addVars(suppliers, customers, vtype=GRB.CONTINUOUS)
y = m.addVars(suppliers, vtype=GRB.BINARY)
obj = gp.quicksum((cost[i][j] * x[i, j] for i in suppliers for j in customers)) + gp.quicksum((fixed_cost[i] * y[i] for i in suppliers))
m.setObjective(obj, GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x[i, j] for i in suppliers)) == demand[j])
for i in suppliers:
    for j in customers:
        m.addConstr(x[i, j] <= demand[j] * y[i])
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(m.ObjVal)
    for v in m.getVars():
        print(v.VarName, v.X)
else:
    print(m.Status)