[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select integer quantities of authorized vehicle configurations for RIVERSIDE_AUTO’s next delivery, maximizing total net benefit (unit benefit minus fixed preparation charges), while respecting resource limits, category quantity bounds, item and category activation fees, incompatibilities, requires dependencies, and awarding bundle bonuses when both options in a bundle are ordered.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation) costs, logical (binary) constraints, and combinatorial bonuses.
3.  **Define Index Sets:** The primary indices are:
    - Items (vehicle configurations): from all rows in the 'item' table (export_06.csv).
    - Categories: from all rows in the 'category' table (export_03.csv).
    - Resources: from all rows in the 'capacity_ledger' and 'usage' tables (export_02.csv, export_09.csv).
    - Bundles: from all rows in the 'bundle' table (export_01.csv).
    - Incompatible pairs: from all rows in the 'incompatible' table (export_05.csv).
    - Requires dependencies: from all rows in the 'requires' table (export_08.csv).
4.  **Define Decision Variables:**
    -   `x[i]` = Integer quantity of item (configuration) `i` to order. Type: GRB.INTEGER. Domain: 0 or between minimum_lot and maximum_order if authorized.
    -   `y[i]` = Binary variable: 1 if item `i` is selected (i.e., x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `z[c]` = Binary variable: 1 if any item in category `c` is selected (i.e., sum of x[i] for items in c > 0), 0 otherwise. Type: GRB.BINARY.
    -   `b[bundle]` = Binary variable: 1 if both items in bundle are selected (i.e., both y[i_a] and y[i_b] are 1), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients:
        -   `unit_benefit_cents` (per unit, from 'item' table/export_06.csv).
        -   `item_fee_cents` (fixed per item if used, from 'item' table/export_06.csv).
        -   `activation_fee_cents` (fixed per category if used, from 'category' table/export_03.csv).
        -   `bonus_cents` (per bundle, from 'bundle' table/export_01.csv).
    -   Constraint coefficients:
        -   `usage` (per item per resource, from 'usage' table/export_09.csv).
        -   `capacity_ledger` (resource limits, sum of 'opening' and 'reservation' per resource, from export_02.csv).
        -   `minimum_lot`, `maximum_order`, `authorized` (from 'item' table/export_06.csv).
        -   `minimum_quantity`, `maximum_quantity` (per category, from 'category' table/export_03.csv).
        -   Incompatible pairs (from 'incompatible' table/export_05.csv).
        -   Requires dependencies (from 'requires' table/export_08.csv).
    -   Index mappings:
        -   Item-to-category mapping (from 'item' table/export_06.csv).
        -   Bundle item pairs (from 'bundle' table/export_01.csv).
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    -   Sum over all items: (unit_benefit_cents[i] * x[i]) 
    -   Minus sum over all items: (item_fee_cents[i] * y[i]) [charged once per item if any units ordered]
    -   Minus sum over all categories: (activation_fee_cents[c] * z[c]) [charged once per category if any item in c is ordered]
    -   Plus sum over all bundles: (bonus_cents[bundle] * b[bundle]) [awarded once per bundle if both items in bundle are ordered]
7.  **Formulate Constraints:**
    -   **Item Authorization and Lot Sizing:**
        -   For each item i: If authorized[i] == 1, x[i] ∈ {0} ∪ [minimum_lot[i], maximum_order[i]] (integer); if authorized[i] == 0, x[i] = 0.
        -   For each item i: y[i] = 1 if x[i] ≥ minimum_lot[i], else y[i] = 0.
        -   For each item i: x[i] ≤ maximum_order[i] * y[i].
    -   **Resource Capacity:**
        -   For each resource r: sum over items i of (usage[i, r] * x[i]) ≤ total_capacity[r], where total_capacity[r] = sum of 'opening' and 'reservation' for r in 'capacity_ledger'.
    -   **Category Quantity Limits:**
        -   For each category c: sum over items i in c of x[i] ≥ minimum_quantity[c] * z[c] (if z[c]=1), and ≤ maximum_quantity[c].
        -   For each category c: z[c] = 1 if any x[i] > 0 for i in c, else 0.
    -   **Incompatibility:**
        -   For each incompatible pair (i, j): y[i] + y[j] ≤ 1 (cannot select both).
    -   **Requires Dependencies:**
        -   For each requires pair (i, prereq): y[i] ≤ y[prereq] (can only select i if prereq is also selected).
        -   Optionally, x[i] > 0 ⇒ x[prereq] > 0 (enforced via y variables).
    -   **Bundle Bonuses:**
        -   For each bundle (i_a, i_b): b[bundle] ≤ y[i_a], b[bundle] ≤ y[i_b], b[bundle] ≥ y[i_a] + y[i_b] - 1 (b[bundle]=1 iff both items are selected).
    -   **Variable Domains:**
        -   x[i]: integer, as above.
        -   y[i], z[c], b[bundle]: binary.
[Abstract Model Plan END]