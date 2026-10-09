[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer order quantities for each authorized product for the CENTRAL_FRESH supermarket, maximizing net return (in USD cents) from the order. The model must account for per-unit benefits, fixed item and category fees, resource usage and capacity, category-level quantity bounds and activation fees, incompatibilities, prerequisites, and bundle bonuses, using only the supplied table rows.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge, resource, and logical (combinatorial) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items (from all rows in the 'item' table, filtered to authorized=1)
    - Categories (from all rows in the 'category' table)
    - Resources (from all rows in the 'resource' columns of 'usage' and 'capacity_ledger' tables)
    - Bundles (from all rows in the 'bundle' table)
    - Incompatible pairs (from all rows in the 'incompatible' table)
    - Prerequisite pairs (from all rows in the 'requires' table)
4.  **Define Decision Variables:**
    -   `x[i]` = Integer number of cases of item `i` to order (for each authorized item). Type: GRB.INTEGER, domain: {0} ∪ [minimum_lot[i], maximum_order[i]].
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
        - Item authorization: 'authorized' (from 'item' table)
        - Minimum/maximum lot/order: 'minimum_lot', 'maximum_order' (from 'item' table)
        - Category min/max: 'minimum_quantity', 'maximum_quantity' (from 'category' table)
        - Incompatibility: pairs from 'incompatible' table
        - Prerequisites: pairs from 'requires' table
    -   Constraint RHS:
        - Resource capacities (after unit conversion)
        - Category min/max quantities
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    - Sum over all items: (unit_benefit_cents[i] * x[i]) 
    - Minus sum over all items: (item_fee_cents[i] * y[i]) [fixed fee if any of item i is ordered]
    - Minus sum over all categories: (activation_fee_cents[c] * z[c]) [fixed fee if any item in category c is ordered]
    - Plus sum over all bundles: (bonus_cents[bundle] * b[bundle]) [bonus if both items in bundle are ordered]
7.  **Formulate Constraints:**
    -   **Item selection and bounds:**
        - For each authorized item i: x[i] = 0, or minimum_lot[i] ≤ x[i] ≤ maximum_order[i]
        - For each unauthorized item: x[i] = 0
        - For all items: x[i] ≥ 0 and integer
        - For all items: y[i] = 1 if x[i] ≥ minimum_lot[i], 0 if x[i] = 0
        - For all items: x[i] ≤ maximum_order[i] * y[i]
        - For all items: x[i] ≥ minimum_lot[i] * y[i]
    -   **Category activation and quantity bounds:**
        - For each category c: z[c] = 1 if any item in c is ordered (i.e., y[i]=1 for any i in c), 0 otherwise
        - For each category c: sum over i in c of x[i] ≥ minimum_quantity[c] * z[c]
        - For each category c: sum over i in c of x[i] ≤ maximum_quantity[c] * z[c]
    -   **Resource capacity constraints:**
        - For each resource r: sum over i of (resource usage per unit of i for r, converted to base units) * x[i] ≤ total available capacity for r (sum of 'opening' and 'reservation' entries, converted to base units)
        - Unit conversions: 1 liter = 1000 ml, 1 hour = 60 minutes, 1 kwh = 1000 wh
    -   **Incompatibility constraints:**
        - For each incompatible pair (i, j): y[i] + y[j] ≤ 1
    -   **Prerequisite constraints:**
        - For each (i requires j): y[i] ≤ y[j] (i can only be ordered if j is also ordered)
    -   **Bundle bonus constraints:**
        - For each bundle (i, j): b[bundle] ≤ y[i], b[bundle] ≤ y[j], b[bundle] ≥ y[i] + y[j] - 1 (b[bundle]=1 iff both y[i]=1 and y[j]=1)
    -   **Variable domains:**
        - x[i]: integer, ≥ 0
        - y[i], z[c], b[bundle]: binary (0 or 1)
[Abstract Model Plan END]