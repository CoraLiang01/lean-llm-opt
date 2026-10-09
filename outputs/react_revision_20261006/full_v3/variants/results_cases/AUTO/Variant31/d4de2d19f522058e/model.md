[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select integer quantities of authorized vehicle configurations for delivery, maximizing total net benefit (unit benefit minus fixed item and category fees, plus bundle bonuses), subject to resource, category, compatibility, and dependency constraints. All data is to be taken directly from the supplied tables.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation) costs, resource constraints, category constraints, incompatibility, and logical (requires) constraints, plus bundle bonuses.
3.  **Define Index Sets:** The primary indices are:
    - Items (vehicle configurations) from the 'item' table (export_06.csv).
    - Categories from the 'category' table (export_03.csv).
    - Resources from the 'capacity_ledger' and 'usage' tables (export_02.csv and export_09.csv).
    - Bundles from the 'bundle' table (export_01.csv).
    - Incompatible pairs from the 'incompatible' table (export_05.csv).
    - Requires pairs from the 'requires' table (export_08.csv).
4.  **Define Decision Variables:**
    -   `x[i]` = Integer quantity of item `i` to order (for each authorized item). Type: GRB.INTEGER.
    -   `y[i]` = 1 if item `i` is selected (i.e., x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `z[c]` = 1 if any item in category `c` is selected, 0 otherwise. Type: GRB.BINARY.
    -   `b[bundle]` = 1 if both items in bundle `bundle` are selected (i.e., both x[i_a] > 0 and x[i_b] > 0), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients:
        -   `unit_benefit_cents` (per unit, from 'item' table).
        -   `item_fee_cents` (fixed per item if used, from 'item' table).
        -   `activation_fee_cents` (fixed per category if used, from 'category' table).
        -   `bonus_cents` (per bundle, from 'bundle' table).
    -   Constraints:
        -   `minimum_lot`, `maximum_order` (per item, from 'item' table).
        -   `authorized` (per item, from 'item' table).
        -   `category` (item-to-category mapping, from 'item' table).
        -   `minimum_quantity`, `maximum_quantity` (per category, from 'category' table).
        -   Resource usage per item (`amount` per resource, from 'usage' table).
        -   Resource capacity (`amount` per resource, from 'capacity_ledger' table: sum of 'opening' and 'reservation' for each resource).
        -   Incompatible pairs (from 'incompatible' table).
        -   Requires pairs (from 'requires' table).
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    -   Sum over all items: (unit_benefit_cents[i] * x[i]) 
    -   Minus sum over all items: (item_fee_cents[i] * y[i]) [charged once per item if any units ordered]
    -   Minus sum over all categories: (activation_fee_cents[c] * z[c]) [charged once per category if any item in category is ordered]
    -   Plus sum over all bundles: (bonus_cents[bundle] * b[bundle]) [awarded once per bundle if both items in bundle are ordered]
7.  **Formulate Constraints:**
    -   **Item Authorization and Bounds:** For each item:
        -   If authorized[i] == 1: minimum_lot[i] * y[i] ≤ x[i] ≤ maximum_order[i] * y[i]; x[i] ≥ 0 integer; y[i] ∈ {0,1}.
        -   If authorized[i] == 0: x[i] = 0; y[i] = 0.
    -   **Category Quantity Limits:** For each category c:
        -   sum over items in c of x[i] ≥ minimum_quantity[c] * z[c]
        -   sum over items in c of x[i] ≤ maximum_quantity[c] * z[c]
        -   z[c] ∈ {0,1}; z[c] = 1 if any x[i] > 0 for i in c, else 0.
    -   **Resource Capacity:** For each resource r:
        -   sum over items: (usage_amount[i,r] * x[i]) ≤ total_capacity[r]
        -   Where total_capacity[r] = sum of 'opening' and 'reservation' for resource r from 'capacity_ledger'.
    -   **Incompatibility:** For each incompatible pair (i, j):
        -   y[i] + y[j] ≤ 1 (cannot select both items in an incompatible pair).
    -   **Requires Dependencies:** For each requires pair (i, prereq):
        -   y[i] ≤ y[prereq] (cannot select i unless prereq is also selected).
        -   Alternatively, x[i] > 0 ⇒ x[prereq] > 0 (enforced via y variables).
    -   **Bundle Bonuses:** For each bundle (i_a, i_b):
        -   b[bundle] ≤ y[i_a]
        -   b[bundle] ≤ y[i_b]
        -   b[bundle] ≥ y[i_a] + y[i_b] - 1
        -   b[bundle] ∈ {0,1} (bonus awarded only if both items are selected).
    -   **Category Activation:** For each category c:
        -   z[c] ≥ y[i] for all items i in c (if any item in c is selected, category is activated).
    -   **Variable Domains:** All x[i] are integer, y[i], z[c], b[bundle] are binary.
[Abstract Model Plan END]