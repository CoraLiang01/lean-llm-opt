[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer order quantities for each authorized bread option to maximize net benefit (in USD cents), subject to storage, staff time, and energy limits, as well as category, incompatibility, prerequisite, and bundle bonus rules. The plan must account for per-unit benefits, fixed item and category activation fees, resource usage, and all logical constraints described in the query.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation) costs, resource constraints, logical (incompatibility, prerequisite, bundle) constraints, and integer variables.
3.  **Define Index Sets:** The primary indices are:
    - Items: All rows in the 'item' table with authorized=1.
    - Categories: All rows in the 'category' table.
    - Resources: All unique resources in the 'usage' and 'capacity_ledger' tables.
    - Incompatible pairs: All rows in the 'incompatible' table.
    - Prerequisite pairs: All rows in the 'requires' table.
    - Bundles: All rows in the 'bundle' table.
4.  **Define Decision Variables:**
    - `q[i]` = Order quantity of item i (authorized items only). Type: GRB.INTEGER, domain: {0} ∪ [minimum_lot[i], maximum_order[i]].
    - `z[i]` = 1 if item i is ordered (q[i] > 0), 0 otherwise. Type: GRB.BINARY.
    - `y[c]` = 1 if any item in category c is ordered, 0 otherwise. Type: GRB.BINARY.
    - `b[p]` = 1 if both items in bundle pair p are ordered (i.e., both q[i]>0 for the pair), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    - Objective coefficients:
        - Per-unit benefit: 'unit_benefit_cents' from 'item' table.
        - Fixed item fee: 'item_fee_cents' from 'item' table.
        - Category activation fee: 'activation_fee_cents' from 'category' table.
        - Bundle bonus: 'bonus_cents' from 'bundle' table.
    - Constraint coefficients:
        - Per-unit resource usage: 'amount' from 'usage' table, joined by item_ref and resource.
        - Resource capacity: sum of 'amount' from 'capacity_ledger' table for each resource (convert all units to base units: ml, minute, wh).
    - Constraint RHS:
        - Category minimum/maximum: 'minimum_quantity', 'maximum_quantity' from 'category' table.
        - Item minimum/maximum: 'minimum_lot', 'maximum_order' from 'item' table.
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
        - Sum over items: (unit_benefit_cents[i] * q[i] - item_fee_cents[i] * z[i])
        - Minus sum over categories: activation_fee_cents[c] * y[c]
        - Plus sum over bundles: bonus_cents[p] * b[p]
7.  **Formulate Constraints:**
    - **Item selection and bounds:** For each authorized item i:
        - q[i] = 0 or minimum_lot[i] ≤ q[i] ≤ maximum_order[i]
        - z[i] = 1 if q[i] > 0, z[i] = 0 if q[i] = 0 (enforced by: q[i] ≥ minimum_lot[i] * z[i], q[i] ≤ maximum_order[i] * z[i])
    - **Category activation:** For each category c:
        - y[c] = 1 if any item in c is ordered (z[i] ≤ y[c] for all i in c; y[c] ≤ sum(z[i] for i in c))
    - **Category quantity bounds:** For each category c:
        - minimum_quantity[c] ≤ sum(q[i] for i in c) ≤ maximum_quantity[c]
    - **Resource constraints:** For each resource r:
        - sum(usage_amount[i,r] * q[i] for all i with usage of r) ≤ total_capacity[r] (convert all units to base units before summing)
    - **Incompatibility:** For each incompatible pair (i, j):
        - z[i] + z[j] ≤ 1
    - **Prerequisite:** For each (i requires j):
        - z[i] ≤ z[j]
    - **Bundle bonuses:** For each bundle pair (i, j):
        - b[p] ≤ z[i], b[p] ≤ z[j], b[p] ≥ z[i] + z[j] - 1 (b[p]=1 iff both z[i]=1 and z[j]=1)
    - **Variable domains:** All q[i] are integer, z[i], y[c], b[p] are binary.
[Abstract Model Plan END]