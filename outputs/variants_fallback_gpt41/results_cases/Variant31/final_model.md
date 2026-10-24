[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select integer quantities of authorized vehicle configurations for RIVERSIDE_AUTO’s next delivery, maximizing total net benefit (unit benefit minus fixed item and category fees, plus bundle bonuses), subject to resource, category, compatibility, and dependency constraints. Only authorized options may be selected, and all constraints and fees must be respected as described.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge and logical (compatibility, dependency, bundle) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items (vehicle configurations) from the 'item' table (item_ref).
    - Categories from the 'category' table (category).
    - Resources from the 'resource' field in 'usage' and 'capacity_ledger' tables.
    - Bundles from the 'bundle' table (pairs of items).
    - Incompatible pairs from the 'incompatible' table.
    - Requires pairs from the 'requires' table.
4.  **Define Decision Variables:**
    -   `q[i]` = Integer quantity of item i selected for delivery (i in Items). Type: GRB.INTEGER.
    -   `z[i]` = 1 if item i is selected (q[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `y[c]` = 1 if any item in category c is selected, 0 otherwise. Type: GRB.BINARY.
    -   `b[bundle]` = 1 if both items in bundle are selected (positive quantity), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Per-item parameters: 'unit_benefit_cents', 'item_fee_cents', 'authorized', 'minimum_lot', 'maximum_order', 'category' (from 'item' table).
    -   Per-category parameters: 'minimum_quantity', 'maximum_quantity', 'activation_fee_cents' (from 'category' table).
    -   Per-resource usage: 'amount' per item per resource (from 'usage' table).
    -   Resource limits: sum of 'amount' for each resource in 'capacity_ledger' table (sum of 'opening' and 'reservation' entries per resource).
    -   Bundle bonuses: 'bonus_cents' for each bundle (from 'bundle' table).
    -   Incompatibility: pairs of items (from 'incompatible' table).
    -   Requires: prerequisite pairs (from 'requires' table).
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    -   Sum over items: (unit_benefit_cents[i] * q[i]) minus (item_fee_cents[i] * z[i])
    -   Minus sum over categories: (activation_fee_cents[c] * y[c])
    -   Plus sum over bundles: (bonus_cents[bundle] * b[bundle])
7.  **Formulate Constraints:**
    -   **Authorization:** For each item i, q[i] = 0 if authorized[i] = 0; else q[i] in {0} or [minimum_lot[i], maximum_order[i]] (enforced via z[i] and bounds).
    -   **Item selection logic:** For each item i, z[i] = 1 iff q[i] > 0; enforce q[i] >= minimum_lot[i] * z[i], q[i] <= maximum_order[i] * z[i], q[i] = 0 if z[i] = 0.
    -   **Category selection logic:** For each category c, y[c] = 1 iff any z[i] = 1 for i in c; enforce z[i] <= y[c] for all i in c, y[c] <= sum(z[i] for i in c).
    -   **Category quantity bounds:** For each category c, sum(q[i] for i in c) >= minimum_quantity[c], sum(q[i] for i in c) <= maximum_quantity[c] (unconditional).
    -   **Resource limits:** For each resource r, sum(usage_amount[i,r] * q[i] for i) <= total_capacity[r] (sum of 'amount' in 'capacity_ledger' for r).
    -   **Incompatibility:** For each incompatible pair (i, j), z[i] + z[j] <= 1.
    -   **Requires dependencies:** For each requires pair (i, prereq), z[i] <= z[prereq].
    -   **Bundle bonuses:** For each bundle (i, j), b[bundle] = 1 iff z[i] = 1 and z[j] = 1; enforce b[bundle] <= z[i], b[bundle] <= z[j], b[bundle] >= z[i] + z[j] - 1.
[Abstract Model Plan END]