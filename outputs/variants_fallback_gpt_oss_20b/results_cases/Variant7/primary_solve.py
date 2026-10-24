CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A distribution network ships goods from source plants to customers through intermediate cross-dock hubs. '
          'Source supplies and customer demands are listed in node_supply_demand.csv, hub throughput capacities are '
          'listed in hub_capacity.csv, and per-unit transportation costs for source-to-hub and hub-to-customer arcs '
          'are listed in arc_costs.csv. The total shipments out of each source may not exceed its available supply; '
          'unused source supply is allowed.\n'
          '\n'
          'Formulate a minimum-cost transshipment model. For each directed arc i-j, define f_ij as the nonnegative '
          'shipment flow on that arc. The objective is to minimize total transportation cost. The model should include '
          'source supply upper-bound constraints, customer demand constraints, flow-balance constraints at each hub, '
          'hub throughput-capacity constraints, and nonnegativity constraints for all arc-flow variables.',
 'relationships': [],
 'route': 'Others',
 'tables': [{'columns': ['Node', 'NodeType', 'Amount'],
             'file_index': 0,
             'file_name': 'node_supply_demand.csv',
             'filters': {},
             'original_rows': 7,
             'records': [{'source_row': 0, 'values': {'Amount': '120', 'Node': 'S1', 'NodeType': 'SourceSupply'}},
                         {'source_row': 1, 'values': {'Amount': '100', 'Node': 'S2', 'NodeType': 'SourceSupply'}},
                         {'source_row': 2, 'values': {'Amount': '90', 'Node': 'S3', 'NodeType': 'SourceSupply'}},
                         {'source_row': 3, 'values': {'Amount': '70', 'Node': 'C1', 'NodeType': 'CustomerDemand'}},
                         {'source_row': 4, 'values': {'Amount': '80', 'Node': 'C2', 'NodeType': 'CustomerDemand'}},
                         {'source_row': 5, 'values': {'Amount': '60', 'Node': 'C3', 'NodeType': 'CustomerDemand'}},
                         {'source_row': 6, 'values': {'Amount': '90', 'Node': 'C4', 'NodeType': 'CustomerDemand'}}],
             'returned_rows': 7,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'},
            {'columns': ['Hub', 'ThroughputCapacity'],
             'file_index': 1,
             'file_name': 'hub_capacity.csv',
             'filters': {},
             'original_rows': 2,
             'records': [{'source_row': 0, 'values': {'Hub': 'H1', 'ThroughputCapacity': '170'}},
                         {'source_row': 1, 'values': {'Hub': 'H2', 'ThroughputCapacity': '160'}}],
             'returned_rows': 2,
             'role': 'descriptive non-unique role',
             'table_id': 'file_1_view_0'},
            {'columns': ['From', 'To', 'Cost'],
             'file_index': 2,
             'file_name': 'arc_costs.csv',
             'filters': {},
             'original_rows': 14,
             'records': [{'source_row': 0, 'values': {'Cost': '2', 'From': 'S1', 'To': 'H1'}},
                         {'source_row': 1, 'values': {'Cost': '6', 'From': 'S1', 'To': 'H2'}},
                         {'source_row': 2, 'values': {'Cost': '4', 'From': 'S2', 'To': 'H1'}},
                         {'source_row': 3, 'values': {'Cost': '3', 'From': 'S2', 'To': 'H2'}},
                         {'source_row': 4, 'values': {'Cost': '7', 'From': 'S3', 'To': 'H1'}},
                         {'source_row': 5, 'values': {'Cost': '2', 'From': 'S3', 'To': 'H2'}},
                         {'source_row': 6, 'values': {'Cost': '3', 'From': 'H1', 'To': 'C1'}},
                         {'source_row': 7, 'values': {'Cost': '4', 'From': 'H1', 'To': 'C2'}},
                         {'source_row': 8, 'values': {'Cost': '7', 'From': 'H1', 'To': 'C3'}},
                         {'source_row': 9, 'values': {'Cost': '8', 'From': 'H1', 'To': 'C4'}},
                         {'source_row': 10, 'values': {'Cost': '8', 'From': 'H2', 'To': 'C1'}},
                         {'source_row': 11, 'values': {'Cost': '6', 'From': 'H2', 'To': 'C2'}},
                         {'source_row': 12, 'values': {'Cost': '3', 'From': 'H2', 'To': 'C3'}},
                         {'source_row': 13, 'values': {'Cost': '4', 'From': 'H2', 'To': 'C4'}}],
             'returned_rows': 14,
             'role': 'descriptive non-unique role',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    data = CSVQA_DATA

    def get_table(table_id):
        for tbl in data['tables']:
            if tbl['table_id'] == table_id:
                return tbl
        raise ValueError(f'Table {table_id} not found in CSVQA_DATA')
    node_tbl = get_table('file_0_view_0')
    supply = {}
    demand = {}
    S = set()
    C = set()
    for rec in node_tbl['records']:
        node = rec['values']['Node']
        node_type = rec['values']['NodeType']
        amount = float(rec['values']['Amount'])
        if node_type == 'SourceSupply':
            S.add(node)
            supply[node] = amount
        elif node_type == 'CustomerDemand':
            C.add(node)
            demand[node] = amount
        else:
            pass
    hub_tbl = get_table('file_1_view_0')
    H = set()
    cap = {}
    for rec in hub_tbl['records']:
        hub = rec['values']['Hub']
        capacity = float(rec['values']['ThroughputCapacity'])
        H.add(hub)
        cap[hub] = capacity
    arc_tbl = get_table('file_2_view_0')
    A = []
    cost = {}
    for rec in arc_tbl['records']:
        frm = rec['values']['From']
        to = rec['values']['To']
        cst = float(rec['values']['Cost'])
        A.append((frm, to))
        cost[frm, to] = cst
    for s in S:
        if s not in supply:
            raise ValueError(f'Missing supply for source {s}')
    for c in C:
        if c not in demand:
            raise ValueError(f'Missing demand for customer {c}')
    for h in H:
        if h not in cap:
            raise ValueError(f'Missing capacity for hub {h}')
    for (i, j) in A:
        if (i, j) not in cost:
            raise ValueError(f'Missing cost for arc {i}->{j}')
    m = gurobipy.Model('Transshipment')
    m.setParam('MIPGap', 1e-06)
    f = m.addVars(A, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gurobipy.quicksum((cost[i, j] * f[i, j] for (i, j) in A)), GRB.MINIMIZE)
    for s in S:
        inflow = gurobipy.quicksum((f[s, h] for h in H if (s, h) in A))
        m.addConstr(inflow <= supply[s], name=f'supply_{s}')
    for c in C:
        outflow = gurobipy.quicksum((f[h, c] for h in H if (h, c) in A))
        m.addConstr(outflow == demand[c], name=f'demand_{c}')
    for h in H:
        inflow = gurobipy.quicksum((f[s, h] for s in S if (s, h) in A))
        outflow = gurobipy.quicksum((f[h, c] for c in C if (h, c) in A))
        m.addConstr(inflow == outflow, name=f'balance_{h}')
        m.addConstr(inflow <= cap[h], name=f'capacity_{h}')
    m.optimize()
    return m
m = solve_problem()