CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'Service centres need to be opened at ten candidate locations SC1‚ÄìSC10 to serve fifteen customers '
          'C1‚ÄìC15. Each centre has a fixed opening cost, and the cost of serving a customer depends on which centre '
          'is chosen. The goal is to minimise the sum of all fixed opening costs and customer‚Äìcentre service costs '
          'while ensuring that every customer is served.\n'
          '    service_centers_fixed_costs.csv gives the fixed opening cost for each centre SC1‚ÄìSC10. '
          'expanded_customer_service_costs.csv gives the per-customer service cost from each centre. Row '
          '‚ÄúCustomer‚Äù is C1‚ÄìC15; each SC column contains the cost to serve that customer from that centre.\n'
          '    Requirements is that, every customer must be assigned to exactly one service centre. A customer may be '
          'assigned only to a centre that is opened. And each opened centre may serve at most 4 customers. Your '
          'decision is to choose open or not open each centre and assign each customer to exactly one opened centre, '
          'so that to minimise total cost = sum of fixed opening costs for all opened centres + sum of service costs '
          'for all customer‚Äìcentre assignments using the costs in the two CSV files named above.',
 'relationships': [],
 'route': 'FLP',
 'tables': [{'columns': ['Customer', 'SC1', 'SC2', 'SC3', 'SC4', 'SC5', 'SC6', 'SC7', 'SC8', 'SC9', 'SC10'],
             'file_index': 0,
             'file_name': 'expanded_customer_service_costs.csv',
             'filters': {},
             'original_rows': 15,
             'records': [{'source_row': 0,
                          'values': {'Customer': 'C1',
                                     'SC1': '15.1',
                                     'SC10': '17.3',
                                     'SC2': '21.2',
                                     'SC3': '14.9',
                                     'SC4': '18.8',
                                     'SC5': '22.9',
                                     'SC6': '16.8',
                                     'SC7': '16.5',
                                     'SC8': '9.4',
                                     'SC9': '16.1'}},
                         {'source_row': 1,
                          'values': {'Customer': 'C2',
                                     'SC1': '13.4',
                                     'SC10': '11.7',
                                     'SC2': '16.3',
                                     'SC3': '20.2',
                                     'SC4': '19.6',
                                     'SC5': '20.9',
                                     'SC6': '22.1',
                                     'SC7': '16.9',
                                     'SC8': '9.4',
                                     'SC9': '13.8'}},
                         {'source_row': 2,
                          'values': {'Customer': 'C3',
                                     'SC1': '15.2',
                                     'SC10': '20.4',
                                     'SC2': '18.8',
                                     'SC3': '14.7',
                                     'SC4': '21.7',
                                     'SC5': '18.1',
                                     'SC6': '18.6',
                                     'SC7': '12.3',
                                     'SC8': '11.2',
                                     'SC9': '11.9'}},
                         {'source_row': 3,
                          'values': {'Customer': 'C4',
                                     'SC1': '16.8',
                                     'SC10': '22.2',
                                     'SC2': '19.1',
                                     'SC3': '18.3',
                                     'SC4': '18.8',
                                     'SC5': '23.1',
                                     'SC6': '15.7',
                                     'SC7': '13.1',
                                     'SC8': '8.6',
                                     'SC9': '15.6'}},
                         {'source_row': 4,
                          'values': {'Customer': 'C5',
                                     'SC1': '13.4',
                                     'SC10': '18.2',
                                     'SC2': '18.6',
                                     'SC3': '20.8',
                                     'SC4': '19.8',
                                     'SC5': '22.1',
                                     'SC6': '18.1',
                                     'SC7': '16.7',
                                     'SC8': '12.1',
                                     'SC9': '11.4'}},
                         {'source_row': 5,
                          'values': {'Customer': 'C6',
                                     'SC1': '12.5',
                                     'SC10': '14.6',
                                     'SC2': '22.5',
                                     'SC3': '15.5',
                                     'SC4': '14.9',
                                     'SC5': '21.6',
                                     'SC6': '21.3',
                                     'SC7': '16.1',
                                     'SC8': '10.7',
                                     'SC9': '11.9'}},
                         {'source_row': 6,
                          'values': {'Customer': 'C7',
                                     'SC1': '12.1',
                                     'SC10': '18.7',
                                     'SC2': '17.1',
                                     'SC3': '19.8',
                                     'SC4': '18.6',
                                     'SC5': '22.1',
                                     'SC6': '20.7',
                                     'SC7': '20.5',
                                     'SC8': '12.2',
                                     'SC9': '15.4'}},
                         {'source_row': 7,
                          'values': {'Customer': 'C8',
                                     'SC1': '12.3',
                                     'SC10': '20.1',
                                     'SC2': '15.7',
                                     'SC3': '17.9',
                                     'SC4': '21.3',
                                     'SC5': '22.7',
                                     'SC6': '15.3',
                                     'SC7': '16.6',
                                     'SC8': '11.4',
                                     'SC9': '14.1'}},
                         {'source_row': 8,
                          'values': {'Customer': 'C9',
                                     'SC1': '16.3',
                                     'SC10': '19.1',
                                     'SC2': '21.3',
                                     'SC3': '17.6',
                                     'SC4': '20.8',
                                     'SC5': '21.8',
                                     'SC6': '17.2',
                                     'SC7': '15.5',
                                     'SC8': '12.6',
                                     'SC9': '19.9'}},
                         {'source_row': 9,
                          'values': {'Customer': 'C10',
                                     'SC1': '12.1',
                                     'SC10': '17.4',
                                     'SC2': '18.7',
                                     'SC3': '14.4',
                                     'SC4': '20.1',
                                     'SC5': '22.7',
                                     'SC6': '14.1',
                                     'SC7': '18.1',
                                     'SC8': '11.4',
                                     'SC9': '18.1'}},
                         {'source_row': 10,
                          'values': {'Customer': 'C11',
                                     'SC1': '16.7',
                                     'SC10': '16.1',
                                     'SC2': '18.7',
                                     'SC3': '15.7',
                                     'SC4': '19.9',
                                     'SC5': '24.2',
                                     'SC6': '18.7',
                                     'SC7': '14.2',
                                     'SC8': '13.1',
                                     'SC9': '14.7'}},
                         {'source_row': 11,
                          'values': {'Customer': 'C12',
                                     'SC1': '11.3',
                                     'SC10': '17.8',
                                     'SC2': '23.8',
                                     'SC3': '15.5',
                                     'SC4': '17.3',
                                     'SC5': '23.2',
                                     'SC6': '17.7',
                                     'SC7': '16.8',
                                     'SC8': '14.5',
                                     'SC9': '15.8'}},
                         {'source_row': 12,
                          'values': {'Customer': 'C13',
                                     'SC1': '15.1',
                                     'SC10': '13.9',
                                     'SC2': '20.5',
                                     'SC3': '15.1',
                                     'SC4': '18.4',
                                     'SC5': '20.6',
                                     'SC6': '17.9',
                                     'SC7': '14.5',
                                     'SC8': '8.5',
                                     'SC9': '14.9'}},
                         {'source_row': 13,
                          'values': {'Customer': 'C14',
                                     'SC1': '8.3',
                                     'SC10': '15.1',
                                     'SC2': '20.7',
                                     'SC3': '14.7',
                                     'SC4': '20.4',
                                     'SC5': '20.6',
                                     'SC6': '14.8',
                                     'SC7': '14.2',
                                     'SC8': '11.5',
                                     'SC9': '14.1'}},
                         {'source_row': 14,
                          'values': {'Customer': 'C15',
                                     'SC1': '12.1',
                                     'SC10': '18.7',
                                     'SC2': '16.3',
                                     'SC3': '16.4',
                                     'SC4': '15.1',
                                     'SC5': '21.3',
                                     'SC6': '19.1',
                                     'SC7': '19.5',
                                     'SC8': '16.7',
                                     'SC9': '11.1'}}],
             'returned_rows': 15,
             'role': 'customer-centre service costs',
             'table_id': 'file_0_view_0'},
            {'columns': ['Service Center', 'Fixed Opening Cost'],
             'file_index': 1,
             'file_name': 'service_centers_fixed_costs.csv',
             'filters': {},
             'original_rows': 10,
             'records': [{'source_row': 0, 'values': {'Fixed Opening Cost': '385.1', 'Service Center': 'SC1'}},
                         {'source_row': 1, 'values': {'Fixed Opening Cost': '546.3', 'Service Center': 'SC2'}},
                         {'source_row': 2, 'values': {'Fixed Opening Cost': '485.2', 'Service Center': 'SC3'}},
                         {'source_row': 3, 'values': {'Fixed Opening Cost': '448.1', 'Service Center': 'SC4'}},
                         {'source_row': 4, 'values': {'Fixed Opening Cost': '324.1', 'Service Center': 'SC5'}},
                         {'source_row': 5, 'values': {'Fixed Opening Cost': '323.9', 'Service Center': 'SC6'}},
                         {'source_row': 6, 'values': {'Fixed Opening Cost': '296.5', 'Service Center': 'SC7'}},
                         {'source_row': 7, 'values': {'Fixed Opening Cost': '522.7', 'Service Center': 'SC8'}},
                         {'source_row': 8, 'values': {'Fixed Opening Cost': '448.7', 'Service Center': 'SC9'}},
                         {'source_row': 9, 'values': {'Fixed Opening Cost': '478.7', 'Service Center': 'SC10'}}],
             'returned_rows': 10,
             'role': 'centre fixed opening costs',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_FRAMES):
    centres_frame = CSVQA_FRAMES['file_1_view_0']
    costs_frame = CSVQA_FRAMES['file_0_view_0']
    I = []
    f = {}
    for (_, row) in centres_frame.iterrows():
        centre = row['Service Center']
        I.append(centre)
        try:
            f[centre] = float(row['Fixed Opening Cost'])
        except Exception:
            raise ValueError(f"Invalid fixed opening cost for centre {centre}: {row['Fixed Opening Cost']}")
    J = []
    for (_, row) in costs_frame.iterrows():
        customer = row['Customer']
        J.append(customer)
    c = {}
    for (_, row) in costs_frame.iterrows():
        customer = row['Customer']
        for centre in I:
            try:
                cij = float(row[centre])
            except Exception:
                raise ValueError(f'Invalid service cost for customer {customer}, centre {centre}: {row[centre]}')
            c[centre, customer] = cij
    if len(I) == 0 or len(J) == 0:
        raise ValueError('No centres or customers found in input data.')
    for centre in I:
        if centre not in f:
            raise ValueError(f'Missing fixed cost for centre {centre}.')
        for customer in J:
            if (centre, customer) not in c:
                raise ValueError(f'Missing service cost for centre {centre}, customer {customer}.')
    m = gp.Model('FLP')
    m.Params.MIPGap = 0.0001
    y_vars = m.addVars(I, vtype=GRB.BINARY, name='')
    x_vars = m.addVars(I, J, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((f[i] * y_vars[i] for i in I)) + gp.quicksum((c[i, j] * x_vars[i, j] for i in I for j in J)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x_vars[i, j] for i in I)) == 1 for j in J), name='')
    m.addConstrs((x_vars[i, j] <= y_vars[i] for i in I for j in J), name='')
    m.addConstrs((gp.quicksum((x_vars[i, j] for j in J)) <= 4 * y_vars[i] for i in I), name='')
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')