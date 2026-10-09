[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer order quantities for each authorized product for the CENTRAL_FRESH supermarket, maximizing net return (in USD cents) from the order. The model must account for per-unit benefits, fixed item and category fees, resource usage and capacity, category quantity bounds, incompatibilities, prerequisites, and bundle bonuses, using only the supplied table rows.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge, resource, and logical (incompatibility, prerequisite, bundle) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items (from 'item' table, filtered to authorized=1)
    - Categories (from 'category' table)
    - Resources (from 'usage' and 'capacity_ledger' tables)
    - Bundles (from 'bundle' table)
    - Incompatible pairs (from 'incompatible' table)
    - Prerequisite pairs (from 'requires' table)
4.  **Define Decision Variables:**
    -   `q[i]` = Integer number of cases of item i to order (i in authorized items). Type: GRB.INTEGER, domain: {0} ∪ [minimum_lot[i], maximum_order[i]].
    -   `z[i]` = 1 if any quantity of item i is ordered, 0 otherwise. Type: GRB.BINARY.
    -   `w[c]` = 1 if any item in category c is ordered, 0 otherwise. Type: GRB.BINARY.
    -   `b[bundle]` = 1 if both items in bundle are ordered with positive quantity, 0 otherwise. Type: GRB.BINARY.
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
        - Category quantity bounds: 'minimum_quantity', 'maximum_quantity' (from 'category' table)
        - Item order bounds: 'minimum_lot', 'maximum_order' (from 'item' table)
        - Incompatibility: pairs from 'incompatible' table
        - Prerequisites: pairs from 'requires' table
    -   Unit conversions: 1 liter = 1000 ml, 1 hour = 60 minutes, 1 kwh = 1000 wh (apply to both usage and capacity)
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    - Sum over items: (unit_benefit_cents[i] * q[i]) 
    - Minus sum over items: (item_fee_cents[i] * z[i]) [fixed fee if any ordered]
    - Minus sum over categories: (activation_fee_cents[c] * w[c]) [fixed fee if any item in category ordered]
    - Plus sum over bundles: (bonus_cents[bundle] * b[bundle]) [bonus if both items in bundle ordered]
7.  **Formulate Constraints:**
    -   **Item Order Bounds:** For each authorized item i: q[i] = 0 or minimum_lot[i] ≤ q[i] ≤ maximum_order[i].
    -   **Item Activation Linking:** For each item i: z[i] = 1 if q[i] ≥ minimum_lot[i], z[i] = 0 if q[i] = 0. (Enforce: q[i] ≥ minimum_lot[i] * z[i], q[i] ≤ maximum_order[i] * z[i])
    -   **Category Activation Linking:** For each category c: w[c] = 1 if any item in c is ordered, 0 otherwise. (Enforce: For all i in c, z[i] ≤ w[c]; and w[c] ≤ sum(z[i] for i in c))
    -   **Category Quantity Bounds:** For each category c: sum(q[i] for i in c) ≥ minimum_quantity[c], sum(q[i] for i in c) ≤ maximum_quantity[c]
    -   **Resource Capacity:** For each resource r: sum over items i of (resource usage per unit of i for r, converted to base units) * q[i] ≤ total available capacity for r (sum of 'amount' in 'capacity_ledger' for r, converted to base units)
    -   **Incompatibility:** For each incompatible pair (i, j): z[i] + z[j] ≤ 1 (cannot order both)
    -   **Prerequisite:** For each (i requires j): z[i] ≤ z[j] (cannot order i unless j is also ordered)
    -   **Bundle Bonus Linking:** For each bundle (i, j): b[bundle] ≤ z[i], b[bundle] ≤ z[j], b[bundle] ≥ z[i] + z[j] - 1 (b[bundle]=1 iff both z[i]=1 and z[j]=1)
    -   **Authorization:** Only items with authorized=1 are included in indices and variables; all others are excluded.
    -   **Integrality:** All q[i] are integer, z[i], w[c], b[bundle] are binary.
[Abstract Model Plan END]