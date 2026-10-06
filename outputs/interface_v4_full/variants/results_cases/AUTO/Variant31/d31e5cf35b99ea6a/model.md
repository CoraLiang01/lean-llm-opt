[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select integer quantities of authorized vehicle configurations for RIVERSIDE_AUTO’s next delivery, maximizing total net benefit (unit benefit minus fixed preparation charges), subject to resource limits, category quantity bounds, item and category activation fees, incompatibility and dependency rules, and bundle bonuses. All data is to be taken directly from the supplied tables.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge, resource, and logical (incompatibility, dependency, bundle) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items (vehicle configurations) from the 'item' table (export_06.csv).
    - Categories from the 'category' table (export_03.csv).
    - Resources from the 'capacity_ledger' and 'usage' tables (export_02.csv, export_09.csv).
    - Bundles from the 'bundle' table (export_01.csv).
    - Incompatible pairs from the 'incompatible' table (export_05.csv).
    - Requires dependencies from the 'requires' table (export_08.csv).
4.  **Define Decision Variables:**
    -   `x[i]` = Integer quantity of item i to order (for each authorized item). Type: GRB.INTEGER.
    -   `y[i]` = 1 if item i is selected (i.e., x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `z[c]` = 1 if any item in category c is selected (i.e., sum of x[i] for items in c > 0), 0 otherwise. Type: GRB.BINARY.
    -   `b[bundle]` = 1 if both items in bundle are selected (i.e., both y[i_a] and y[i_b] are 1), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients:
        -   'unit_benefit_cents' (per unit, from 'item' table).
        -   'item_fee_cents' (fixed per item if any units ordered, from 'item' table).
        -   'activation_fee_cents' (fixed per category if any item in category is ordered, from 'category' table).
        -   'bonus_cents' (per bundle, from 'bundle' table).
    -   Constraint coefficients:
        -   'usage' table: per-unit resource usage for each item/resource.
        -   'capacity_ledger' table: total available resource (sum of 'opening' and 'reservation' for each resource).
        -   'minimum_lot', 'maximum_order' (per item, from 'item' table).
        -   'minimum_quantity', 'maximum_quantity' (per category, from 'category' table).
        -   'authorized' (per item, from 'item' table).
        -   Incompatible pairs (from 'incompatible' table).
        -   Requires dependencies (from 'requires' table).
    -   Constraint RHS:
        -   Resource limits (from 'capacity_ledger').
        -   Category quantity bounds (from 'category').
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    -   Sum over items: (unit_benefit_cents[i] * x[i]) 
    -   Minus sum over items: (item_fee_cents[i] * y[i]) [charged once per item if any units ordered]
    -   Minus sum over categories: (activation_fee_cents[c] * z[c]) [charged once per category if any item in c is ordered]
    -   Plus sum over bundles: (bonus_cents[bundle] * b[bundle]) [awarded once per bundle if both items in bundle are selected]
7.  **Formulate Constraints:**
    -   **Item Authorization and Lot Sizing:**
        -   For each item i: If authorized[i] == 1, x[i] ∈ {0} ∪ [minimum_lot[i], maximum_order[i]] (integer); if authorized[i] == 0, x[i] = 0.
        -   For each item i: y[i] = 1 if x[i] ≥ 1, else 0. Enforce with: x[i] ≥ minimum_lot[i] * y[i], x[i] ≤ maximum_order[i] * y[i].
    -   **Category Quantity Bounds:**
        -   For each category c: sum of x[i] over items in c ≥ minimum_quantity[c] * z[c], sum ≤ maximum_quantity[c].
        -   z[c] = 1 if any x[i] > 0 for items in c, else 0.
    -   **Resource Limits:**
        -   For each resource r: sum over items i of (usage[i, r] * x[i]) ≤ total available for r (sum of 'opening' and 'reservation' in 'capacity_ledger').
    -   **Incompatibility:**
        -   For each incompatible pair (i, j): y[i] + y[j] ≤ 1 (cannot select both).
    -   **Requires Dependencies:**
        -   For each (i, prereq): y[i] ≤ y[prereq] (can only select i if prereq is also selected).
        -   Optionally, x[i] ≥ 1 ⇒ x[prereq] ≥ 1 (if positive quantity of i, must have positive quantity of prereq).
    -   **Bundle Bonuses:**
        -   For each bundle (i_a, i_b): b[bundle] ≤ y[i_a], b[bundle] ≤ y[i_b], b[bundle] ≥ y[i_a] + y[i_b] - 1 (b[bundle] = 1 iff both items are selected).
    -   **Variable Domains:**
        -   x[i] ∈ {0} ∪ [minimum_lot[i], maximum_order[i]] (integer, only for authorized items).
        -   y[i], z[c], b[bundle] ∈ {0,1}.
[Abstract Model Plan END]