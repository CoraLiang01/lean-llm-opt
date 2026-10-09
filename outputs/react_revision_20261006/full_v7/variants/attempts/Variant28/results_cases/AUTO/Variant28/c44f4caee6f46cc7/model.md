[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the integer number of cases to order for each authorized product for the CENTRAL_FRESH supermarket, maximizing net return (in USD cents) from the order. The model must account for per-unit benefits, fixed item and category activation fees, resource usage and capacity, category-level quantity bounds, item-level order bounds, incompatibilities, prerequisites, and bundle bonuses.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation) costs, resource constraints, logical (incompatibility/prerequisite) constraints, and bundle bonuses.
3.  **Define Index Sets:** The primary indices are:
    - Items (authorized products, from the 'item' table, filtered where authorized=1)
    - Categories (from the 'category' table)
    - Resources (from the 'usage' and 'capacity_ledger' tables)
    - Bundles (from the 'bundle' table)
    - Incompatible pairs (from the 'incompatible' table)
    - Prerequisite pairs (from the 'requires' table)
4.  **Define Decision Variables:**
    -   `x[i]` = Number of cases of item `i` to order. Type: GRB.INTEGER, domain: {0} ∪ [minimum_lot[i], maximum_order[i]] for each authorized item.
    -   `y[i]` = 1 if any quantity of item `i` is ordered, 0 otherwise. Type: GRB.BINARY.
    -   `z[c]` = 1 if any item in category `c` is ordered, 0 otherwise. Type: GRB.BINARY.
    -   `b[bundle]` = 1 if both items in bundle `bundle` are ordered in positive quantity, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients:
        - Per-unit benefit: 'unit_benefit_cents' (from 'item' table)
        - Fixed item fee: 'item_fee_cents' (from 'item' table)
        - Category activation fee: 'activation_fee_cents' (from 'category' table)
        - Bundle bonus: 'bonus_cents' (from 'bundle' table)
    -   Constraint coefficients:
        - Resource usage per unit: 'amount' and 'unit' (from 'usage' table, per item/resource)
        - Resource capacity: sum of 'amount' (converted to base units) from 'capacity_ledger' table, per resource
        - Category membership: 'category' (from 'item' table)
        - Item-level order bounds: 'minimum_lot', 'maximum_order' (from 'item' table)
        - Category-level quantity bounds: 'minimum_quantity', 'maximum_quantity' (from 'category' table)
        - Incompatibilities: pairs from 'incompatible' table
        - Prerequisites: pairs from 'requires' table
        - Authorization: 'authorized' (from 'item' table; only authorized=1 items are eligible)
    -   Constraint RHS:
        - Resource capacities (from 'capacity_ledger')
        - Category quantity bounds (from 'category')
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    - Sum over all items: (unit_benefit_cents[i] * x[i]) 
    - Minus sum over all items: (item_fee_cents[i] * y[i]) [fee paid once per item if any ordered]
    - Minus sum over all categories: (activation_fee_cents[c] * z[c]) [fee paid once per category if any item in c is ordered]
    - Plus sum over all bundles: (bonus_cents[bundle] * b[bundle]) [bonus earned once per bundle if both items in bundle are ordered in positive quantity]
7.  **Formulate Constraints:**
    -   **Item Order Bounds:** For each authorized item i:
        - x[i] = 0, or minimum_lot[i] ≤ x[i] ≤ maximum_order[i]
        - y[i] = 1 if x[i] ≥ minimum_lot[i], y[i] = 0 if x[i] = 0
        - Enforce: x[i] ≥ minimum_lot[i] * y[i], x[i] ≤ maximum_order[i] * y[i]
    -   **Category Quantity Bounds:** For each category c:
        - Let S_c = set of authorized items in category c
        - sum_{i in S_c} x[i] ≥ minimum_quantity[c]
        - sum_{i in S_c} x[i] ≤ maximum_quantity[c]
        - z[c] = 1 if any x[i] > 0 for i in S_c, else 0
        - Enforce: For all i in S_c, y[i] ≤ z[c]; sum_{i in S_c} y[i] ≥ z[c]
    -   **Resource Capacity Constraints:** For each resource r:
        - For each item i, get per-unit usage u[i,r] (from 'usage'), convert units to base (ml, wh, minute) as needed
        - sum_{i} (u[i,r] * x[i]) ≤ total available capacity for r (sum of 'amount' in 'capacity_ledger' for r, after unit conversion)
    -   **Incompatibility Constraints:** For each incompatible pair (i, j):
        - y[i] + y[j] ≤ 1 (cannot order both items in the pair)
    -   **Prerequisite Constraints:** For each (i, prereq) in 'requires':
        - y[i] ≤ y[prereq] (cannot order i unless also order its prerequisite)
    -   **Bundle Bonus Constraints:** For each bundle (a, b):
        - b[bundle] ≤ y[a], b[bundle] ≤ y[b], b[bundle] ≥ y[a] + y[b] - 1
        - Only bundles where both items are authorized can be triggered (if either is unauthorized, b[bundle]=0)
    -   **Authorization Constraint:** Only items with authorized=1 are eligible; all variables for unauthorized items are fixed at zero.
    -   **Variable Domains:** x[i] ∈ {0} ∪ [minimum_lot[i], maximum_order[i]] (integers); y[i], z[c], b[bundle] ∈ {0,1}
[Abstract Model Plan END]