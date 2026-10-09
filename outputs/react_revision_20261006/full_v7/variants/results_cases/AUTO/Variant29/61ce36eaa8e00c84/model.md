[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer order quantities for each authorized bread option to maximize net benefit (in USD cents), considering per-unit benefits, fixed item and category activation fees, bundle bonuses, and subject to storage, staff time, energy, category, incompatibility, and prerequisite constraints.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation) costs, resource constraints, logical (incompatibility/prerequisite) constraints, and bundle bonuses.
3.  **Define Index Sets:** The primary indices are:
    - Items (bread options) — from all rows in the 'item' table where authorized=1.
    - Categories — from all rows in the 'category' table.
    - Resources — from all rows in the 'usage' and 'capacity_ledger' tables.
    - Incompatible pairs — from all rows in the 'incompatible' table.
    - Prerequisite pairs — from all rows in the 'requires' table.
    - Bundles — from all rows in the 'bundle' table.
4.  **Define Decision Variables:**
    -   `x[i]` = Number of units of item i to order (for each authorized item). Type: GRB.INTEGER, domain: {0} ∪ [minimum_lot[i], maximum_order[i]].
    -   `y[i]` = 1 if item i is ordered (x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `z[c]` = 1 if any item in category c is ordered, 0 otherwise. Type: GRB.BINARY.
    -   `b[bundle]` = 1 if both items in bundle are ordered (and both are authorized), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Per-unit benefit: 'unit_benefit_cents' from 'item' table.
    -   Item fixed fee: 'item_fee_cents' from 'item' table.
    -   Category activation fee: 'activation_fee_cents' from 'category' table.
    -   Bundle bonus: 'bonus_cents' from 'bundle' table.
    -   Minimum/maximum order: 'minimum_lot', 'maximum_order' from 'item' table.
    -   Category min/max quantity: 'minimum_quantity', 'maximum_quantity' from 'category' table.
    -   Resource usage per unit: 'amount' and 'unit' from 'usage' table (per item/resource).
    -   Resource capacity: sum of 'amount' (opening + reservation) and 'unit' from 'capacity_ledger' table (per resource).
    -   Incompatibility: pairs from 'incompatible' table.
    -   Prerequisites: pairs from 'requires' table.
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    -   Sum over items: (unit_benefit_cents[i] * x[i]) 
    -   Minus sum over items: (item_fee_cents[i] * y[i]) [if item ordered]
    -   Minus sum over categories: (activation_fee_cents[c] * z[c]) [if any item in category ordered]
    -   Plus sum over bundles: (bonus_cents[bundle] * b[bundle]) [if both items in bundle are ordered and authorized]
7.  **Formulate Constraints:**
    -   **Item Order Bounds:** For each authorized item i: x[i] = 0, or minimum_lot[i] ≤ x[i] ≤ maximum_order[i].
    -   **Item Activation Linking:** For each authorized item i: y[i] = 1 if x[i] > 0, y[i] = 0 if x[i] = 0. (Enforced via: x[i] ≥ minimum_lot[i] * y[i], x[i] ≤ maximum_order[i] * y[i])
    -   **Category Activation Linking:** For each category c: z[c] = 1 if any item in c is ordered (i.e., y[i]=1 for any i in c), z[c] = 0 otherwise.
    -   **Category Quantity Bounds:** For each category c: sum over i in c of x[i] ≥ minimum_quantity[c], sum over i in c of x[i] ≤ maximum_quantity[c].
    -   **Resource Constraints:** For each resource r: sum over items i of (resource usage per unit[i,r] * x[i]) ≤ total available capacity[r], after converting all units to a common base (e.g., liters, minutes, wh).
    -   **Incompatibility Constraints:** For each incompatible pair (i, j): y[i] + y[j] ≤ 1.
    -   **Prerequisite Constraints:** For each (i requires j): y[i] ≤ y[j].
    -   **Bundle Bonus Linking:** For each bundle (i, j): b[bundle] ≤ y[i], b[bundle] ≤ y[j], b[bundle] ≥ y[i] + y[j] - 1; and only allow b[bundle]=1 if both i and j are authorized.
    -   **Variable Domains:** x[i] ∈ {0} ∪ [minimum_lot[i], maximum_order[i]] (integers); y[i], z[c], b[bundle] ∈ {0,1}.
[Abstract Model Plan END]