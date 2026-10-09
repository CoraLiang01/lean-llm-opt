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
 'route': 'TP',
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
             'role': 'node supply and demand',
             'table_id': 'file_0_view_0'},
            {'columns': ['Hub', 'ThroughputCapacity'],
             'file_index': 1,
             'file_name': 'hub_capacity.csv',
             'filters': {},
             'original_rows': 2,
             'records': [{'source_row': 0, 'values': {'Hub': 'H1', 'ThroughputCapacity': '170'}},
                         {'source_row': 1, 'values': {'Hub': 'H2', 'ThroughputCapacity': '160'}}],
             'returned_rows': 2,
             'role': 'hub throughput capacity',
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
             'role': 'arc transportation costs',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_FRAMES):
    import pandas as pd
    node_df = CSVQA_FRAMES['file_0_view_0']
    hub_df = CSVQA_FRAMES['file_1_view_0']
    arc_df = CSVQA_FRAMES['file_2_view_0']
    sources = node_df[node_df['NodeType'].str.casefold() == 'sourcesupply']['Node'].tolist()
    customers = node_df[node_df['NodeType'].str.casefold() == 'customerdemand']['Node'].tolist()
    hubs = hub_df['Hub'].tolist()
    nodes = sources + hubs + customers
    arc_keys = []
    arc_cost = {}
    for (idx, row) in arc_df.iterrows():
        i = row['From']
        j = row['To']
        arc_keys.append((i, j))
        try:
            arc_cost[i, j] = float(row['Cost'])
        except Exception:
            raise ValueError(f"Non-numeric cost for arc ({i},{j}): {row['Cost']}")
    source_supply = {}
    for (idx, row) in node_df.iterrows():
        if row['NodeType'].casefold() == 'sourcesupply':
            try:
                source_supply[row['Node']] = float(row['Amount'])
            except Exception:
                raise ValueError(f"Non-numeric supply for source {row['Node']}: {row['Amount']}")
    customer_demand = {}
    for (idx, row) in node_df.iterrows():
        if row['NodeType'].casefold() == 'customerdemand':
            try:
                customer_demand[row['Node']] = float(row['Amount'])
            except Exception:
                raise ValueError(f"Non-numeric demand for customer {row['Node']}: {row['Amount']}")
    hub_capacity = {}
    for (idx, row) in hub_df.iterrows():
        try:
            hub_capacity[row['Hub']] = float(row['ThroughputCapacity'])
        except Exception:
            raise ValueError(f"Non-numeric throughput for hub {row['Hub']}: {row['ThroughputCapacity']}")
    for (i, j) in arc_keys:
        if i not in nodes or j not in nodes:
            raise ValueError(f'Arc ({i},{j}) references unknown node(s)')
    m = gp.Model('DistributionNetworkTransshipment')
    flow_vars = m.addVars(arc_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((arc_cost[i, j] * flow_vars[i, j] for (i, j) in arc_keys)), GRB.MINIMIZE)
    for s in sources:
        outgoing = [(i, j) for (i, j) in arc_keys if i == s]
        m.addConstr(gp.quicksum((flow_vars[i, j] for (i, j) in outgoing)) <= source_supply[s], name=f'supply_{s}')
    for c in customers:
        incoming = [(i, j) for (i, j) in arc_keys if j == c]
        m.addConstr(gp.quicksum((flow_vars[i, j] for (i, j) in incoming)) >= customer_demand[c], name=f'demand_{c}')
    for h in hubs:
        incoming = [(i, j) for (i, j) in arc_keys if j == h]
        outgoing = [(i, j) for (i, j) in arc_keys if i == h]
        m.addConstr(gp.quicksum((flow_vars[i, j] for (i, j) in incoming)) == gp.quicksum((flow_vars[i, j] for (i, j) in outgoing)), name=f'flowbal_{h}')
    for h in hubs:
        incoming = [(i, j) for (i, j) in arc_keys if j == h]
        m.addConstr(gp.quicksum((flow_vars[i, j] for (i, j) in incoming)) <= hub_capacity[h], name=f'cap_{h}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem(CSVQA_FRAMES)