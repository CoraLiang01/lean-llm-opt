[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer order quantities for each authorized bread option to maximize net benefit (total unit benefit plus bundle bonuses, minus item and category activation fees), subject to storage, staff time, and energy resource limits, category-level minimum and maximum order quantities, item-level lot and order bounds, incompatibility and prerequisite requirements, and bundle bonus eligibility.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation) and logical (incompatibility, prerequisite, bundle) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items: all rows in the 'item' table with authorized=1.
    - Categories: all rows in the 'category' table.
    - Resources: all unique resources in the 'usage' and 'capacity_ledger' tables.
    - Incompatible pairs: all rows in the 'incompatible' table.
    - Prerequisite pairs: all rows in the 'requires' table.
    - Bundles: all rows in the 'bundle' table.
4.  **Define Decision Variables:**
    - `q[i]` = integer quantity ordered of item i (authorized items only). Type: GRB.INTEGER, domain: {0} ∪ [minimum_lot[i], maximum_order[i]].
    - `y[i]` = 1 if item i is ordered (q[i] > 0), 0 otherwise. Type: GRB.BINARY.
    - `z[c]` = 1 if any item in category c is ordered, 0 otherwise. Type: GRB.BINARY.
    - `b[bundle]` = 1 if both items in bundle are ordered and authorized, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    - Objective coefficients:
        - `unit_benefit_cents[i]` and `item_fee_cents[i]` from 'item' table.
        - `activation_fee_cents[c]` from 'category' table.
        - `bonus_cents[bundle]` from 'bundle' table.
    - Constraint coefficients:
        - `amount[i,r]` and `unit[i,r]` from 'usage' table (per-unit resource use by item).
        - `amount[r,entry]` and `unit[r,entry]` from 'capacity_ledger' table (resource capacity, sum over entries).
    - Constraint RHS:
        - Resource limits: sum of 'capacity_ledger' entries per resource, converted to common units.
        - Category minimum and maximum: `minimum_quantity[c]`, `maximum_quantity[c]` from 'category' table.
        - Item minimum and maximum: `minimum_lot[i]`, `maximum_order[i]` from 'item' table.
        - Incompatibility: pairs from 'incompatible' table.
        - Prerequisites: pairs from 'requires' table.
6.  **Formulate Objective:** Maximize total net benefit in cents:
    - Sum over items: `unit_benefit_cents[i] * q[i] - item_fee_cents[i] * y[i]`
    - Minus sum over categories: `activation_fee_cents[c] * z[c]`
    - Plus sum over bundles: `bonus_cents[bundle] * b[bundle]`
7.  **Formulate Constraints:**
    - Resource limits: For each resource r, sum over items i of (per-unit usage of r by i, converted to the resource's unit) times q[i] ≤ total available capacity for r (sum of 'capacity_ledger' entries for r, converted to same unit).
    - Item order bounds: For each item i, q[i] = 0 or minimum_lot[i] ≤ q[i] ≤ maximum_order[i].
    - Item activation: For each item i, y[i] = 1 iff q[i] > 0; enforce q[i] ≥ y[i] and q[i] ≤ maximum_order[i] * y[i].
    - Category activation: For each category c, z[c] = 1 iff any item in c is ordered; enforce y[i] ≤ z[c] for all i in c, and z[c] ≤ sum(y[i] for i in c).
    - Category quantity bounds: For each category c, sum(q[i] for i in c) ≥ minimum_quantity[c] and ≤ maximum_quantity[c].
    - Incompatibility: For each incompatible pair (i,j), y[i] + y[j] ≤ 1.
    - Prerequisites: For each (i requires j), y[i] ≤ y[j].
    - Bundle bonuses: For each bundle (i,j), b[bundle] ≤ y[i], b[bundle] ≤ y[j], b[bundle] ≥ y[i] + y[j] - 1; only allow b[bundle]=1 if both items are authorized.
[Abstract Model Plan END]