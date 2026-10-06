[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer order quantities for each authorized product for the CENTRAL_FRESH supermarket, maximizing net return (in USD cents) from the order. The model must account for per-unit benefits, fixed item and category fees, resource usage and capacity, category quantity bounds, incompatibilities, prerequisites, and bundle bonuses, using only the items and relationships explicitly listed in the supplied tables.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge, resource, logical (incompatibility/prerequisite), and bundle bonus constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items: All item rows from the item table (export_06.csv) with authorized=1.
    - Categories: All category rows from the category table (export_03.csv).
    - Resources: All resource rows from the usage and capacity_ledger tables (export_09.csv, export_02.csv).
    - Incompatible pairs: All pairs from the incompatible table (export_05.csv).
    - Prerequisite pairs: All pairs from the requires table (export_08.csv).
    - Bundles: All rows from the bundle table (export_01.csv).
4.  **Define Decision Variables:**
    -   `x[i]` = Integer number of cases of item i to order (for each authorized item). Type: GRB.INTEGER, domain: {0} ∪ [minimum_lot[i], maximum_order[i]].
    -   `y[i]` = Binary variable: 1 if any quantity of item i is ordered, 0 otherwise. Type: GRB.BINARY.
    -   `z[c]` = Binary variable: 1 if any item in category c is ordered, 0 otherwise. Type: GRB.BINARY.
    -   `b[bundle]` = Binary variable: 1 if both items in bundle are ordered in positive quantity, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients:
        - Per-unit benefit: 'unit_benefit_cents' from item table (export_06.csv).
        - Fixed item fee: 'item_fee_cents' from item table (export_06.csv).
        - Category activation fee: 'activation_fee_cents' from category table (export_03.csv).
        - Bundle bonus: 'bonus_cents' from bundle table (export_01.csv).
    -   Constraint coefficients:
        - Resource usage per unit: 'amount' and 'unit' from usage table (export_09.csv), mapped to each item and resource.
        - Resource capacity: sum of 'amount' (with sign) from capacity_ledger table (export_02.csv), converted to base units (ml, wh, minute).
        - Category membership: 'category' field in item and category tables.
        - Minimum/maximum order per item: 'minimum_lot', 'maximum_order' from item table.
        - Category min/max: 'minimum_quantity', 'maximum_quantity' from category table.
        - Incompatibility: pairs from incompatible table (export_05.csv).
        - Prerequisites: pairs from requires table (export_08.csv).
        - Authorization: 'authorized' field in item table (only items with authorized=1 are eligible).
    -   All units for resource usage and capacity must be converted to base units (1 liter = 1000 ml, 1 hour = 60 minutes, 1 kwh = 1000 wh).
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    - Sum over all items: (unit_benefit_cents[i] * x[i]) 
    - Minus sum over all items: (item_fee_cents[i] * y[i]) [fee paid once per item if ordered]
    - Minus sum over all categories: (activation_fee_cents[c] * z[c]) [fee paid once per category if any item in c is ordered]
    - Plus sum over all bundles: (bonus_cents[bundle] * b[bundle]) [bonus paid once per bundle if both items in bundle are ordered]
7.  **Formulate Constraints:**
    -   **Item Order Bounds:** For each authorized item i:
        - x[i] = 0, or x[i] ∈ [minimum_lot[i], maximum_order[i]] (enforced via y[i]: x[i] ≥ minimum_lot[i] * y[i], x[i] ≤ maximum_order[i] * y[i], x[i] ≥ 0, integer).
    -   **Category Quantity Bounds:** For each category c:
        - sum_{i in c} x[i] ≥ minimum_quantity[c] * z[c]
        - sum_{i in c} x[i] ≤ maximum_quantity[c] * z[c]
        - z[c] = 1 if any x[i] > 0 for i in c, else 0.
    -   **Resource Capacity:** For each resource r:
        - sum_{i} (resource_usage_per_unit[i, r] * x[i]) ≤ total_capacity[r]
        - All resource usage and capacity must be in the same base units.
    -   **Incompatibility:** For each incompatible pair (i, j):
        - y[i] + y[j] ≤ 1 (cannot order both items in the pair).
    -   **Prerequisite:** For each (i, prereq) in requires:
        - y[i] ≤ y[prereq] (cannot order i unless its prerequisite is also ordered).
    -   **Bundle Bonuses:** For each bundle (a, b):
        - b[bundle] ≤ y[a], b[bundle] ≤ y[b], b[bundle] ≥ y[a] + y[b] - 1 (bonus only if both items are ordered in positive quantity and both are authorized).
    -   **Authorization:** Only items with authorized=1 are eligible; all variables for unauthorized items are fixed at zero.
    -   **Variable Domains:** x[i] ∈ {0} ∪ [minimum_lot[i], maximum_order[i]] (integer), y[i], z[c], b[bundle] ∈ {0,1}.
[Abstract Model Plan END]