[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the integer order quantities for each authorized product (item) for the CENTRAL_FRESH supermarket, maximizing net return (in USD cents) from the order. The model must account for per-unit benefits, fixed item and category fees, resource usage and capacity, category-level quantity bounds and activation fees, incompatibilities, prerequisites, and bundle bonuses, using only the supplied item rows.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge, resource, and logical (incompatibility, prerequisite, bundle) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items (from export_06.csv, filtered to authorized=1)
    - Categories (from export_03.csv)
    - Resources (from export_09.csv and export_02.csv)
    - Bundles (from export_01.csv)
    - Incompatible pairs (from export_05.csv)
    - Prerequisite pairs (from export_08.csv)
4.  **Define Decision Variables:**
    -   `x[i]` = Integer number of cases of item i to order (i in authorized items). Type: GRB.INTEGER.
    -   `y[i]` = 1 if any quantity of item i is ordered, 0 otherwise (i in authorized items). Type: GRB.BINARY.
    -   `z[c]` = 1 if any item in category c is ordered, 0 otherwise (c in categories). Type: GRB.BINARY.
    -   `b[bundle]` = 1 if both items in bundle are ordered in positive quantity, 0 otherwise (bundle in bundles). Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients:
        - Per-unit benefit: 'unit_benefit_cents' (export_06.csv)
        - Fixed item fee: 'item_fee_cents' (export_06.csv)
        - Category activation fee: 'activation_fee_cents' (export_03.csv)
        - Bundle bonus: 'bonus_cents' (export_01.csv)
    -   Constraint coefficients:
        - Resource usage per unit: 'amount' and 'unit' (export_09.csv, usage table)
        - Resource capacity: sum of 'amount' (export_02.csv, capacity_ledger table), with unit conversions as needed
        - Category membership: 'category' (export_06.csv)
        - Category quantity bounds: 'minimum_quantity', 'maximum_quantity' (export_03.csv)
        - Incompatibilities: pairs from export_05.csv
        - Prerequisites: pairs from export_08.csv
    -   Constraint RHS:
        - Resource capacities (after unit conversion)
        - Category quantity bounds
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    - Sum over items: (unit_benefit_cents[i] * x[i]) 
    - Minus sum over items: (item_fee_cents[i] * y[i]) [fee paid once per item if ordered]
    - Minus sum over categories: (activation_fee_cents[c] * z[c]) [fee paid once per category if any item in c is ordered]
    - Plus sum over bundles: (bonus_cents[bundle] * b[bundle]) [bonus earned once per bundle if both items ordered]
7.  **Formulate Constraints:**
    -   **Item selection and bounds:**
        - For each authorized item i: x[i] = 0, or minimum_lot[i] ≤ x[i] ≤ maximum_order[i]
        - For each i: y[i] = 1 if x[i] ≥ 1, else y[i] = 0 (linking: x[i] ≥ y[i], x[i] ≤ maximum_order[i] * y[i])
    -   **Category quantity bounds and activation:**
        - For each category c: sum_{i in c} x[i] ≥ minimum_quantity[c] * z[c]
        - For each category c: sum_{i in c} x[i] ≤ maximum_quantity[c] * z[c]
        - For each i in c: y[i] ≤ z[c] (if any item in c is ordered, z[c]=1)
    -   **Resource capacity:**
        - For each resource r: sum_{i} (resource_usage_per_unit[i,r] * x[i]) ≤ total_capacity[r]
        - All units must be converted to a common base (e.g., kwh→wh, liter→ml, hour→minute) before summing
    -   **Incompatibility:**
        - For each incompatible pair (i, j): y[i] + y[j] ≤ 1 (cannot order both)
    -   **Prerequisite:**
        - For each (i requires j): y[i] ≤ y[j] (cannot order i unless j is also ordered)
    -   **Bundle bonuses:**
        - For each bundle (i, j): b[bundle] ≤ y[i], b[bundle] ≤ y[j], b[bundle] ≥ y[i] + y[j] - 1 (bonus only if both ordered)
        - Only bundles where both items are authorized can be triggered
    -   **Variable domains:**
        - x[i] ∈ {0} ∪ [minimum_lot[i], maximum_order[i]] (integers)
        - y[i], z[c], b[bundle] ∈ {0,1}
[Abstract Model Plan END]