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
    node_frame = CSVQA_FRAMES['file_0_view_0']
    sources = []
    customers = []
    s_param = {}
    d_param = {}
    for (_, row) in node_frame.iterrows():
        node = row['Node']
        ntype = row['NodeType']
        amount = float(row['Amount'])
        if ntype.casefold() == 'sourcesupply':
            sources.append(node)
            s_param[node] = amount
        elif ntype.casefold() == 'customerdemand':
            customers.append(node)
            d_param[node] = amount
    hub_frame = CSVQA_FRAMES['file_1_view_0']
    hubs = []
    u_param = {}
    for (_, row) in hub_frame.iterrows():
        hub = row['Hub']
        hubs.append(hub)
        u_param[hub] = float(row['ThroughputCapacity'])
    arc_frame = CSVQA_FRAMES['file_2_view_0']
    arcs = []
    c_param = {}
    for (_, row) in arc_frame.iterrows():
        i = row['From']
        j = row['To']
        arcs.append((i, j))
        c_param[i, j] = float(row['Cost'])
    source_out_arcs = {i: [] for i in sources}
    for (i, j) in arcs:
        if i in sources:
            source_out_arcs[i].append((i, j))
    customer_in_arcs = {j: [] for j in customers}
    for (i, j) in arcs:
        if j in customers:
            customer_in_arcs[j].append((i, j))
    hub_in_arcs = {h: [] for h in hubs}
    hub_out_arcs = {h: [] for h in hubs}
    for (i, j) in arcs:
        if j in hubs:
            hub_in_arcs[j].append((i, j))
        if i in hubs:
            hub_out_arcs[i].append((i, j))
    m = gp.Model('min_cost_transshipment')
    m.Params.MIPGap = 0.0001
    flow_vars = m.addVars(arcs, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((c_param[i, j] * flow_vars[i, j] for (i, j) in arcs)), GRB.MINIMIZE)
    for i in sources:
        m.addConstr(gp.quicksum((flow_vars[i, j] for (i, j) in source_out_arcs[i])) <= s_param[i], name='supply_' + i)
    for j in customers:
        m.addConstr(gp.quicksum((flow_vars[i, j] for (i, j) in customer_in_arcs[j])) >= d_param[j], name='demand_' + j)
    for h in hubs:
        m.addConstr(gp.quicksum((flow_vars[i, h] for (i, h2) in hub_in_arcs[h])) == gp.quicksum((flow_vars[h, j] for (h2, j) in hub_out_arcs[h])), name='hub_balance_' + h)
    for h in hubs:
        m.addConstr(gp.quicksum((flow_vars[i, h] for (i, h2) in hub_in_arcs[h])) <= u_param[h], name='hub_capacity_' + h)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem(CSVQA_FRAMES)