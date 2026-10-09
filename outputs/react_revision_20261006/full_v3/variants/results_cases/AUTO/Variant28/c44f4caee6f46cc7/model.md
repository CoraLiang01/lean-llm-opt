[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer order quantities for each authorized product (item) for the CENTRAL_FRESH supermarket, maximizing net return (in USD cents) from the order. The model must account for per-unit benefits, fixed item and category fees, resource usage and capacity, category-level quantity bounds and activation fees, incompatibilities, prerequisite requirements, and bundle bonuses.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge, resource, and logical (combinatorial) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items (authorized products, from the 'item' table, filtered where authorized=1)
    - Categories (from the 'category' table)
    - Resources (from the 'usage' and 'capacity_ledger' tables)
    - Bundles (from the 'bundle' table)
    - Incompatible pairs (from the 'incompatible' table)
    - Prerequisite pairs (from the 'requires' table)
4.  **Define Decision Variables:**
    -   `x[i]` = Integer number of cases of item i to order (for each authorized item). Type: GRB.INTEGER.
    -   `z[i]` = 1 if any quantity of item i is ordered (i.e., x[i] ≥ minimum_lot), 0 otherwise. Type: GRB.BINARY.
    -   `w[c]` = 1 if any item in category c is ordered (i.e., sum of x[i] for items in c ≥ 1), 0 otherwise. Type: GRB.BINARY.
    -   `b[bundle]` = 1 if both items in bundle are ordered with positive quantity, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients:
        - Per-unit benefit: 'unit_benefit_cents' (from 'item' table)
        - Fixed item fee: 'item_fee_cents' (from 'item' table, paid once per item if ordered)
        - Category activation fee: 'activation_fee_cents' (from 'category' table, paid once per category if any item in category is ordered)
        - Bundle bonus: 'bonus_cents' (from 'bundle' table, earned once per bundle if both items are ordered)
    -   Constraint coefficients:
        - Resource usage per unit: 'amount' and 'unit' (from 'usage' table, per item per resource)
        - Resource capacity: sum of 'amount' (from 'capacity_ledger' table, per resource, after unit conversion)
        - Item minimum/maximum order: 'minimum_lot', 'maximum_order' (from 'item' table)
        - Category minimum/maximum total quantity: 'minimum_quantity', 'maximum_quantity' (from 'category' table)
        - Incompatibility: pairs of items (from 'incompatible' table)
        - Prerequisite: pairs of items (from 'requires' table)
    -   Constraint RHS (limits):
        - Resource capacities (after summing and unit conversion)
        - Category quantity bounds
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    - Sum over all items: (unit_benefit_cents[i] * x[i]) 
    - Minus sum over all ordered items: (item_fee_cents[i] * z[i])
    - Minus sum over all activated categories: (activation_fee_cents[c] * w[c])
    - Plus sum over all triggered bundles: (bonus_cents[bundle] * b[bundle])
7.  **Formulate Constraints:**
    -   **Item Order Bounds:** For each authorized item i: x[i] = 0, or minimum_lot[i] ≤ x[i] ≤ maximum_order[i]. (Enforced via x[i] ≥ minimum_lot[i] * z[i], x[i] ≤ maximum_order[i] * z[i], x[i] ≥ 0, integer)
    -   **Item Activation:** z[i] = 1 if x[i] ≥ minimum_lot[i], z[i] = 0 if x[i] = 0.
    -   **Category Activation:** For each category c, w[c] = 1 if any item in c is ordered (sum over i in c of z[i] ≥ 1), w[c] = 0 otherwise.
    -   **Category Quantity Bounds:** For each category c, sum over i in c of x[i] ≥ minimum_quantity[c], and ≤ maximum_quantity[c].
    -   **Resource Capacity:** For each resource r, sum over all items i of (resource usage per unit for i in r, after converting units to match capacity_ledger) * x[i] ≤ total available capacity for r (sum of 'amount' in capacity_ledger for r, after unit conversion).
    -   **Incompatibility:** For each incompatible pair (i, j), z[i] + z[j] ≤ 1 (cannot order both).
    -   **Prerequisite:** For each (i requires j), z[i] ≤ z[j] (cannot order i unless j is also ordered).
    -   **Bundle Bonus:** For each bundle (i, j), b[bundle] ≤ z[i], b[bundle] ≤ z[j], b[bundle] ≥ z[i] + z[j] - 1 (b[bundle]=1 iff both z[i]=1 and z[j]=1, and both items are authorized).
    -   **Authorization:** Only items with authorized=1 are included in the model; all others have x[i]=0, z[i]=0.
    -   **Variable Types:** x[i] ∈ {0} ∪ [minimum_lot[i], maximum_order[i]] (integer), z[i], w[c], b[bundle] ∈ {0,1}.
[Abstract Model Plan END]