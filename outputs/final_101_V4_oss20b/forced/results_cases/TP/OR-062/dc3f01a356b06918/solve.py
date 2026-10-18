CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The Iowa Department of Commerce requires that any store selling alcohol in bottled form for off-premises '
          'consumption must hold a Class “E” liquor license, a typical arrangement for most state liquor regulatory '
          'authorities. All alcohol sales from stores registered with the Iowa Department of Commerce are recorded in '
          'the department’s system, which is publicly released as open data by the State of Iowa. Several suppliers '
          'located in different cities can provide the necessary liquor products to these licensed stores. Each '
          'supplier incurs a fixed cost when starting operations, with the fixed cost data provided in '
          '“fixed_cost.csv.” The Department needs to source a unit of each liquor product for the stores from these '
          'suppliers. For each product, the transportation cost per unit from each supplier to each store is recorded '
          'in “transportation_costs.csv.” Additionally, each store has a specific demand for these products, which is '
          'provided in “demand.csv.” The objective is to determine which suppliers to activate so that the demand for '
          'all liquor products across all licensed stores is met while minimizing the total cost. The decision '
          'variables y_i are binary, indicating whether a supplier is operational (open). The decision variables '
          'x_{ij} represent the quantity of goods that each store S_j sources from supplier F_i.',
 'relationships': [],
 'route': 'TP',
 'tables': [{'columns': ['Customer', 'demand'],
             'file_index': 0,
             'file_name': 'demand.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0, 'values': {'Customer': 'Customer_1', 'demand': '2397'}},
                         {'source_row': 1, 'values': {'Customer': 'Customer_2', 'demand': '1889'}},
                         {'source_row': 2, 'values': {'Customer': 'Customer_3', 'demand': '2518'}},
                         {'source_row': 3, 'values': {'Customer': 'Customer_4', 'demand': '3218'}},
                         {'source_row': 4, 'values': {'Customer': 'Customer_5', 'demand': '1813'}}],
             'returned_rows': 5,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['Unnamed: 0', 'fixed_costs'],
             'file_index': 1,
             'file_name': 'fixed_cost.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0, 'values': {'Unnamed: 0': 'MOUNT AYR', 'fixed_costs': '96.58'}},
                         {'source_row': 1, 'values': {'Unnamed: 0': 'WAUKEE', 'fixed_costs': '94.06'}},
                         {'source_row': 2, 'values': {'Unnamed: 0': 'WAVERLY', 'fixed_costs': '94.37'}},
                         {'source_row': 3, 'values': {'Unnamed: 0': 'PELLA', 'fixed_costs': '82.88'}},
                         {'source_row': 4, 'values': {'Unnamed: 0': 'DES MOINES', 'fixed_costs': '94.95999999999999'}}],
             'returned_rows': 5,
             'role': 'file_1',
             'table_id': 'file_1_view_0'},
            {'columns': ['Unnamed: 0', 'CLARINDA', 'FORT MADISON', 'SIOUX CITY', 'TOLEDO', 'BANCROFT'],
             'file_index': 2,
             'file_name': 'transportation_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0,
                          'values': {'BANCROFT': '1685.53',
                                     'CLARINDA': '694.6799999999999',
                                     'FORT MADISON': '17.48',
                                     'SIOUX CITY': '20.07',
                                     'TOLEDO': '199.02',
                                     'Unnamed: 0': 'MOUNT AYR'}},
                         {'source_row': 1,
                          'values': {'BANCROFT': '90.69',
                                     'CLARINDA': '15.13',
                                     'FORT MADISON': '1.5',
                                     'SIOUX CITY': '1.43',
                                     'TOLEDO': '27.88',
                                     'Unnamed: 0': 'WAUKEE'}},
                         {'source_row': 2,
                          'values': {'BANCROFT': '78.73',
                                     'CLARINDA': '2.34',
                                     'FORT MADISON': '349.34',
                                     'SIOUX CITY': '246.6',
                                     'TOLEDO': '41.3',
                                     'Unnamed: 0': 'WAVERLY'}},
                         {'source_row': 3,
                          'values': {'BANCROFT': '38.93',
                                     'CLARINDA': '1181.6',
                                     'FORT MADISON': '1458.53',
                                     'SIOUX CITY': '1646.36',
                                     'TOLEDO': '1924.55',
                                     'Unnamed: 0': 'PELLA'}},
                         {'source_row': 4,
                          'values': {'BANCROFT': '103.84',
                                     'CLARINDA': '1030.8',
                                     'FORT MADISON': '43.48',
                                     'SIOUX CITY': '932.4299999999999',
                                     'TOLEDO': '55.39',
                                     'Unnamed: 0': 'DES MOINES'}}],
             'returned_rows': 5,
             'role': 'file_2',
             'table_id': 'file_2_view_0'}],
 'validation': {'fallback_reason': "Plan omitted explicit query subset filters: ['E']",
                'planner_errors': ["Plan omitted explicit query subset filters: ['E']"],
                'status': 'FALLBACK_FULL_DATA'}}
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
data = CSVQA_DATA
demand_records = data['tables'][0]['records']
demand_df = pd.DataFrame([rec['values'] for rec in demand_records])
demand_df['demand'] = pd.to_numeric(demand_df['demand'])
customers = demand_df['Customer'].tolist()
demand_dict = dict(zip(demand_df['Customer'], demand_df['demand']))
fixed_records = data['tables'][1]['records']
fixed_df = pd.DataFrame([rec['values'] for rec in fixed_records])
fixed_df['fixed_costs'] = pd.to_numeric(fixed_df['fixed_costs'])
suppliers = fixed_df['Unnamed: 0'].tolist()
fixed_dict = dict(zip(fixed_df['Unnamed: 0'], fixed_df['fixed_costs']))
trans_records = data['tables'][2]['records']
trans_df = pd.DataFrame([rec['values'] for rec in trans_records])
cost_cols = [c for c in trans_df.columns if c not in ['Unnamed: 0']]
for col in cost_cols:
    trans_df[col] = pd.to_numeric(trans_df[col])
cost_dict = {}
for _, row in trans_df.iterrows():
    supplier = row['Unnamed: 0']
    cost_dict[supplier] = {cust: row[cust] for cust in cost_cols}
M = sum(demand_dict.values())
m = gp.Model('supplier_selection')
m.setParam(GRB.Param.MIPGap, 1e-06)
y = {i: m.addVar(vtype=GRB.BINARY) for i in suppliers}
x = {(i, j): m.addVar(vtype=GRB.CONTINUOUS) for i in suppliers for j in customers}
obj = gp.quicksum((fixed_dict[i] * y[i] for i in suppliers)) + gp.quicksum((cost_dict[i][j] * x[i, j] for i in suppliers for j in customers))
m.setObjective(obj, GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x[i, j] for i in suppliers)) >= demand_dict[j])
for i in suppliers:
    for j in customers:
        m.addConstr(x[i, j] <= M * y[i])
m.optimize()
if m.status == GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for var in m.getVars():
        print(f'{var.VarName} = {var.X}')
else:
    print(f'Solver status: {m.status}')