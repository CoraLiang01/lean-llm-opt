[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer order quantities for each authorized bread option to maximize net benefit (in USD cents), considering per-unit benefits, fixed item and category fees, resource limits (storage, staff time, energy), category quantity bounds, incompatibilities, prerequisites, and bundle bonuses, using only the items and constraints described in the supplied tables.
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
    -   `y[i]` = Binary variable: 1 if item i is ordered (x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `z[c]` = Binary variable: 1 if any item in category c is ordered, 0 otherwise. Type: GRB.BINARY.
    -   `b[bundle]` = Binary variable: 1 if both items in bundle are ordered (and both are authorized), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Per-item parameters: minimum_lot, maximum_order, unit_benefit_cents, item_fee_cents, category (from 'item' table).
    -   Category parameters: minimum_quantity, maximum_quantity, activation_fee_cents (from 'category' table).
    -   Resource usage: amount and unit per item per resource (from 'usage' table).
    -   Resource capacities: sum of 'amount' per resource from 'capacity_ledger' table (convert all units to base units: ml, minute, wh).
    -   Incompatibilities: pairs of items (from 'incompatible' table).
    -   Prerequisites: pairs of items (from 'requires' table).
    -   Bundles: pairs of items and bonus_cents (from 'bundle' table).
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    -   Sum over items: (unit_benefit_cents[i] * x[i]) 
    -   Minus sum over items: (item_fee_cents[i] * y[i]) [fixed fee if item is ordered]
    -   Minus sum over categories: (activation_fee_cents[c] * z[c]) [fixed fee if any item in category is ordered]
    -   Plus sum over bundles: (bonus_cents[bundle] * b[bundle]) [bonus if both items in bundle are ordered and authorized]
7.  **Formulate Constraints:**
    -   **Item Order Bounds:** For each authorized item i: x[i] = 0 or minimum_lot[i] ≤ x[i] ≤ maximum_order[i]; enforce x[i]=0 if not ordered.
    -   **Item Activation Linking:** For each item i: y[i] = 1 if x[i] > 0, else 0. (x[i] ≥ minimum_lot[i] * y[i]; x[i] ≤ maximum_order[i] * y[i])
    -   **Category Activation Linking:** For each category c: z[c] = 1 if any item in c is ordered, else 0. (z[c] ≥ y[i] for all i in c)
    -   **Category Quantity Bounds:** For each category c: sum over i in c of x[i] ≥ minimum_quantity[c] * z[c]; sum over i in c of x[i] ≤ maximum_quantity[c] * z[c]
    -   **Resource Constraints:** For each resource r: sum over items i of (usage_amount[i,r] * x[i] * unit_conversion_factor) ≤ total_capacity[r] (with all units converted to base units: ml, minute, wh)
    -   **Incompatibility Constraints:** For each incompatible pair (i, j): y[i] + y[j] ≤ 1
    -   **Prerequisite Constraints:** For each (i requires j): y[i] ≤ y[j]
    -   **Bundle Bonus Linking:** For each bundle (i, j): b[bundle] ≤ y[i], b[bundle] ≤ y[j], b[bundle] ≥ y[i] + y[j] - 1; only define b[bundle] if both i and j are authorized
    -   **Authorization:** Only items with authorized=1 may be ordered (x[i]=0, y[i]=0 for unauthorized items)
    -   **Integrality:** x[i] ∈ {0} ∪ [minimum_lot[i], maximum_order[i]] (integers); y[i], z[c], b[bundle] ∈ {0,1}
[Abstract Model Plan END]