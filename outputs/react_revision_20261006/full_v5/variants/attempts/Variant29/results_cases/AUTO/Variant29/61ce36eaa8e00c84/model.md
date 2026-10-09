[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer order quantities for each authorized bread option to maximize net benefit (in USD cents), considering per-unit benefits, fixed item and category fees, resource limits (storage, staff time, energy), category quantity bounds, incompatibilities, prerequisites, and bundle bonuses, using only the items and constraints described in the supplied tables.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation) costs, resource constraints, logical (incompatibility/prerequisite) constraints, and bundle bonuses.
3.  **Define Index Sets:** The primary indices are:
    - Items: All item_refs from the 'item' table (filtered to authorized=1).
    - Categories: All categories from the 'category' table.
    - Resources: All resources from the 'usage' and 'capacity_ledger' tables.
    - Incompatible pairs: All (item_a, item_b) from the 'incompatible' table.
    - Prerequisite pairs: All (item_ref, prerequisite_ref) from the 'requires' table.
    - Bundles: All (item_a, item_b) from the 'bundle' table.
4.  **Define Decision Variables:**
    -   `x[i]` = Integer quantity ordered of item i (i in authorized items). Type: GRB.INTEGER, domain: {0} ∪ [minimum_lot[i], maximum_order[i]].
    -   `y[i]` = 1 if item i is ordered (x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `z[c]` = 1 if any item in category c is ordered, 0 otherwise. Type: GRB.BINARY.
    -   `b[bundle]` = 1 if both items in bundle are ordered (and both authorized), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients:
        -   Per-unit benefit: 'unit_benefit_cents' from 'item' table.
        -   Fixed item fee: 'item_fee_cents' from 'item' table.
        -   Category activation fee: 'activation_fee_cents' from 'category' table.
        -   Bundle bonus: 'bonus_cents' from 'bundle' table.
    -   Constraint coefficients:
        -   Per-unit resource usage: 'amount' from 'usage' table (with 'unit' conversion as needed).
    -   Constraint RHS:
        -   Resource capacities: sum of 'amount' from 'capacity_ledger' table for each resource (convert units as needed).
        -   Category min/max: 'minimum_quantity', 'maximum_quantity' from 'category' table.
6.  **Formulate Objective:** Maximize total net benefit in cents:
    -   Sum over items: (unit_benefit_cents[i] * x[i]) 
    -   Minus sum over items: (item_fee_cents[i] * y[i]) 
    -   Minus sum over categories: (activation_fee_cents[c] * z[c])
    -   Plus sum over bundles: (bonus_cents[bundle] * b[bundle])
7.  **Formulate Constraints:**
    -   **Item selection and bounds:** For each authorized item i:
        -   x[i] = 0, or minimum_lot[i] ≤ x[i] ≤ maximum_order[i]
        -   y[i] = 1 if x[i] > 0, y[i] = 0 if x[i] = 0 (enforced via x[i] ≥ minimum_lot[i] * y[i], x[i] ≤ maximum_order[i] * y[i])
    -   **Category activation:** For each category c:
        -   z[c] = 1 if any item in c is ordered (z[c] ≥ y[i] for all i in c)
    -   **Category quantity bounds:** For each category c:
        -   minimum_quantity[c] ≤ sum_{i in c} x[i] ≤ maximum_quantity[c]
    -   **Resource constraints:** For each resource r:
        -   sum over items i: (resource_usage[i,r] * x[i]) ≤ total_capacity[r]
        -   All units must be converted to match (e.g., kwh→wh, hour→minute, liter→ml)
    -   **Incompatibility:** For each incompatible pair (i, j):
        -   y[i] + y[j] ≤ 1
    -   **Prerequisite:** For each (i, prereq) in 'requires':
        -   y[i] ≤ y[prereq]
    -   **Bundle bonuses:** For each bundle (i, j):
        -   b[bundle] ≤ y[i], b[bundle] ≤ y[j], b[bundle] ≥ y[i] + y[j] - 1
        -   Only bundles where both i and j are authorized can be activated (otherwise, b[bundle]=0)
    -   **Variable domains:** x[i] ∈ {0} ∪ [minimum_lot[i], maximum_order[i]] (integers), y[i], z[c], b[bundle] ∈ {0,1}
[Abstract Model Plan END]