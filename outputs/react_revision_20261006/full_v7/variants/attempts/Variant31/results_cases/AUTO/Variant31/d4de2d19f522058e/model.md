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
    -   `z[i]` = 1 if item `i` is selected (i.e., x[i] ≥ 1), 0 otherwise. Type: GRB.BINARY.
    -   `w[c]` = 1 if any item in category `c` is selected (i.e., sum over i in c of z[i] ≥ 1), 0 otherwise. Type: GRB.BINARY.
    -   `b[bundle]` = 1 if both items in bundle are selected (i.e., z[item_a] = 1 and z[item_b] = 1), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients:
        -   `unit_benefit_cents` (per unit, from 'item' table).
        -   `item_fee_cents` (fixed per item if any units ordered, from 'item' table).
        -   `activation_fee_cents` (fixed per category if any item in category is ordered, from 'category' table).
        -   `bonus_cents` (per bundle, from 'bundle' table).
    -   Constraint coefficients:
        -   `usage` (per item per resource, from 'usage' table).
        -   `capacity_ledger` (total available per resource, from 'capacity_ledger' table: sum of 'opening' and 'reservation' for each resource).
        -   `minimum_lot`, `maximum_order` (per item, from 'item' table).
        -   `authorized` (per item, from 'item' table).
        -   `minimum_quantity`, `maximum_quantity` (per category, from 'category' table).
        -   Incompatible pairs (from 'incompatible' table).
        -   Requires pairs (from 'requires' table).
    -   Index mappings:
        -   Items to categories (from 'item' table).
        -   Items to resources (from 'usage' table).
        -   Bundles: pairs of items (from 'bundle' table).
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    -   Sum over all items: (unit_benefit_cents[i] * x[i]) 
    -   Minus sum over all items: (item_fee_cents[i] * z[i]) [charged once per item if any units ordered]
    -   Minus sum over all categories: (activation_fee_cents[c] * w[c]) [charged once per category if any item in category is ordered]
    -   Plus sum over all bundles: (bonus_cents[bundle] * b[bundle]) [awarded once per bundle if both items are selected]
7.  **Formulate Constraints:**
    -   **Item Authorization and Quantity Bounds:**
        -   For each item i: If authorized[i] = 1, then minimum_lot[i] ≤ x[i] ≤ maximum_order[i]; if authorized[i] = 0, then x[i] = 0.
        -   For each item i: z[i] = 1 if x[i] ≥ 1, z[i] = 0 if x[i] = 0 (enforced via x[i] ≥ z[i], x[i] ≤ maximum_order[i] * z[i]).
    -   **Resource Capacity Constraints:**
        -   For each resource r: sum over items i of (usage[i, r] * x[i]) ≤ total_capacity[r], where total_capacity[r] = sum of 'opening' and 'reservation' for r from 'capacity_ledger'.
    -   **Category Quantity and Activation Constraints:**
        -   For each category c: minimum_quantity[c] ≤ sum over items i in c of x[i] ≤ maximum_quantity[c].
        -   For each category c: w[c] = 1 if any z[i] = 1 for i in c, w[c] = 0 otherwise (enforced via z[i] ≤ w[c] for all i in c, and w[c] ≤ sum over i in c of z[i]).
    -   **Incompatibility Constraints:**
        -   For each incompatible pair (i, j): z[i] + z[j] ≤ 1 (cannot select both items in an incompatible pair).
    -   **Requires (Dependency) Constraints:**
        -   For each requires pair (i, prereq): z[i] ≤ z[prereq] (can only select i if prerequisite is also selected).
    -   **Bundle Bonus Constraints:**
        -   For each bundle (item_a, item_b): b[bundle] ≤ z[item_a], b[bundle] ≤ z[item_b], b[bundle] ≥ z[item_a] + z[item_b] - 1 (b[bundle] = 1 iff both items are selected).
    -   **Variable Domains:**
        -   x[i] ∈ {0} ∪ [minimum_lot[i], maximum_order[i]] for authorized items; x[i] = 0 for unauthorized items.
        -   z[i], w[c], b[bundle] ∈ {0, 1}.
[Abstract Model Plan END]