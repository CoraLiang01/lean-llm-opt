[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer order quantities for each authorized bread option to maximize net benefit (in USD cents), considering per-unit benefits, fixed item and category fees, bundle bonuses, resource limits (storage, staff time, energy), category quantity bounds, incompatibilities, and prerequisite requirements. Only authorized options may be ordered, and all constraints and bonuses must be respected.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation) costs, resource constraints, logical (incompatibility/prerequisite) constraints, and bundle bonuses.
3.  **Define Index Sets:** The primary indices are:
    - Items: All item_refs from the 'item' table (filtered to authorized=1).
    - Categories: All categories from the 'category' table.
    - Resources: All resources from the 'usage' and 'capacity_ledger' tables.
    - Incompatible pairs: All (item_a, item_b) from the 'incompatible' table.
    - Prerequisite pairs: All (item_ref, prerequisite_ref) from the 'requires' table.
    - Bundles: All (item_a, item_b) from the 'bundle' table.
4.  **Define Decision Variables:**
    -   `x[i]` = Number of units of item i to order (for each authorized item). Type: GRB.INTEGER, domain: {0} ∪ [minimum_lot[i], maximum_order[i]].
    -   `y[i]` = 1 if any units of item i are ordered (i.e., x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `z[c]` = 1 if any item in category c is ordered (i.e., sum over i in c of x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `b[bundle]` = 1 if both items in bundle are ordered (i.e., x[item_a] > 0 and x[item_b] > 0 and both authorized), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients:
        -   Per-unit benefit: 'unit_benefit_cents' from 'item' table.
        -   Fixed item fee: 'item_fee_cents' from 'item' table.
        -   Category activation fee: 'activation_fee_cents' from 'category' table.
        -   Bundle bonus: 'bonus_cents' from 'bundle' table.
    -   Constraint coefficients:
        -   Per-unit resource usage: 'amount' and 'unit' from 'usage' table (per item/resource).
        -   Resource capacities: sum of 'amount' (with sign) from 'capacity_ledger' table, converted to consistent units (ml, minutes, wh).
        -   Category quantity bounds: 'minimum_quantity', 'maximum_quantity' from 'category' table.
        -   Item order bounds: 'minimum_lot', 'maximum_order' from 'item' table.
        -   Incompatibilities: pairs from 'incompatible' table.
        -   Prerequisites: pairs from 'requires' table.
        -   Authorization: 'authorized' from 'item' table (only authorized=1 items are eligible).
6.  **Formulate Objective:** Maximize total net benefit in cents:
    -   Sum over all items: (unit_benefit_cents[i] * x[i]) 
    -   Minus sum over all items: (item_fee_cents[i] * y[i]) [fixed fee if any units ordered]
    -   Minus sum over all categories: (activation_fee_cents[c] * z[c]) [fixed fee if any item in category ordered]
    -   Plus sum over all bundles: (bonus_cents[bundle] * b[bundle]) [bonus if both items in bundle are ordered and both are authorized]
7.  **Formulate Constraints:**
    -   **Item Order Bounds:** For each authorized item i:
        -   x[i] = 0, or minimum_lot[i] ≤ x[i] ≤ maximum_order[i].
        -   Enforced via: x[i] ≥ y[i] * minimum_lot[i], x[i] ≤ y[i] * maximum_order[i], y[i] ∈ {0,1}.
    -   **Resource Limits:** For each resource r:
        -   sum over items i of (resource_usage[i,r] * x[i]) ≤ total_capacity[r], with all units converted (e.g., liters to ml, hours to minutes, kwh to wh).
    -   **Category Quantity Bounds:** For each category c:
        -   minimum_quantity[c] ≤ sum over items i in c of x[i] ≤ maximum_quantity[c].
        -   z[c] = 1 if any x[i] > 0 for i in c, else 0.
    -   **Category Activation Fee Linking:** For each category c:
        -   For all i in c: x[i] ≤ M * z[c], where M is a large constant (e.g., sum of maximum_order for c).
    -   **Incompatibility:** For each incompatible pair (i, j):
        -   y[i] + y[j] ≤ 1 (cannot order both).
    -   **Prerequisite:** For each (i, prereq) in 'requires':
        -   y[i] ≤ y[prereq] (cannot order i unless also order its prerequisite).
    -   **Bundle Bonus:** For each bundle (i, j):
        -   b[bundle] ≤ y[i], b[bundle] ≤ y[j], b[bundle] ≥ y[i] + y[j] - 1.
        -   Only bundles where both i and j are authorized can be activated.
    -   **Authorization:** Only items with authorized=1 are included in variables and constraints; all others have x[i]=y[i]=0.
    -   **Variable Domains:** x[i] ∈ {0} ∪ [minimum_lot[i], maximum_order[i]] (integers); y[i], z[c], b[bundle] ∈ {0,1}.
[Abstract Model Plan END]