[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer order quantities for each authorized bread option to maximize net benefit (in USD cents), subject to storage, staff time, and energy limits, as well as category, incompatibility, prerequisite, and bundle bonus rules. The model must account for per-unit benefits, fixed item and category fees, resource usage, and bundle bonuses, using only the items and relationships specified in the supplied tables.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation) costs, resource constraints, logical (incompatibility/prerequisite) constraints, and bundle bonuses.
3.  **Define Index Sets:** The primary indices are:
    - Items (authorized bread options, from the 'item' table with authorized=1)
    - Categories (from the 'category' table)
    - Resources (from the 'usage' and 'capacity_ledger' tables)
    - Incompatible pairs (from the 'incompatible' table)
    - Prerequisite pairs (from the 'requires' table)
    - Bundles (from the 'bundle' table)
4.  **Define Decision Variables:**
    -   `x[i]` = Integer quantity ordered of item i (for each authorized item). Type: GRB.INTEGER, domain: {0} ∪ [minimum_lot[i], maximum_order[i]].
    -   `y[i]` = 1 if any quantity of item i is ordered, 0 otherwise (for each authorized item). Type: GRB.BINARY.
    -   `z[c]` = 1 if any item in category c is ordered, 0 otherwise. Type: GRB.BINARY.
    -   `b[bundle]` = 1 if both items in bundle are ordered with positive quantity, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Per-item parameters: 'unit_benefit_cents', 'item_fee_cents', 'minimum_lot', 'maximum_order', 'category' (from 'item' table, filtered to authorized=1).
    -   Per-category parameters: 'minimum_quantity', 'maximum_quantity', 'activation_fee_cents' (from 'category' table).
    -   Resource usage: 'amount' and 'unit' per item/resource (from 'usage' table, for authorized items).
    -   Resource capacities: sum of 'amount' per resource (from 'capacity_ledger' table, convert units as needed).
    -   Incompatibility: pairs of items (from 'incompatible' table).
    -   Prerequisites: pairs of items (from 'requires' table).
    -   Bundle bonuses: 'bonus_cents' for each bundle (from 'bundle' table).
6.  **Formulate Objective:** Maximize total net benefit in USD cents, defined as:
    -   Sum over items: (unit_benefit_cents[i] * x[i]) 
    -   Minus sum over items: (item_fee_cents[i] * y[i]) [fixed fee if any of item i is ordered]
    -   Minus sum over categories: (activation_fee_cents[c] * z[c]) [fixed fee if any item in category c is ordered]
    -   Plus sum over bundles: (bonus_cents[bundle] * b[bundle]) [bonus if both items in bundle are ordered with positive quantity]
7.  **Formulate Constraints:**
    -   **Item order bounds:** For each authorized item i: x[i] = 0 or minimum_lot[i] ≤ x[i] ≤ maximum_order[i].
    -   **Item activation linking:** For each authorized item i: y[i] = 1 if x[i] ≥ minimum_lot[i], y[i] = 0 if x[i] = 0. (Enforced via: x[i] ≥ minimum_lot[i] * y[i], x[i] ≤ maximum_order[i] * y[i])
    -   **Category activation linking:** For each category c: z[c] = 1 if any item in c is ordered, 0 otherwise. (z[c] ≥ y[i] for all i in c; z[c] ≤ sum over i in c of y[i])
    -   **Category quantity bounds:** For each category c: sum over i in c of x[i] ≥ minimum_quantity[c], sum over i in c of x[i] ≤ maximum_quantity[c].
    -   **Resource constraints:** For each resource r: sum over authorized items i of (usage_amount[i,r] * x[i]) ≤ total_capacity[r] (convert all units to common base: ml, minutes, wh).
    -   **Incompatibility:** For each incompatible pair (i, j): y[i] + y[j] ≤ 1.
    -   **Prerequisite:** For each (i requires j): y[i] ≤ y[j].
    -   **Bundle bonuses:** For each bundle (i, j): b[bundle] ≤ y[i], b[bundle] ≤ y[j], b[bundle] ≥ y[i] + y[j] - 1. Only bundles where both items are authorized can be triggered.
    -   **Variable domains:** x[i] ∈ {0} ∪ [minimum_lot[i], maximum_order[i]] (integers); y[i], z[c], b[bundle] ∈ {0,1}.
[Abstract Model Plan END]