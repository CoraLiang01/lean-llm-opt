[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer quantities of authorized air-conditioner models to place in each storage area at FC_EAST_HVAC, maximizing net value (in USD cents). The plan must respect area volume limits, item and category constraints, incompatibilities, prerequisites, and bundle bonuses, using only the supplied item rows and their parameters.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation) and logical (incompatibility, prerequisite, bundle) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items: Each authorized item_ref from the supplied item tables (export_06.csv and export_07.csv).
    - Categories: Each category from the category table (export_03.csv).
    - Resources/Areas: Each location_id/resource from the capacity_ledger and usage tables (export_02.csv, export_10.csv, export_11.csv).
    - Bundles: Each bundle pair from the bundle table (export_01.csv).
    - Incompatibilities: Each incompatible item pair from the incompatible table (export_05.csv).
    - Prerequisites: Each requires pair from the requires table (export_09.csv).
4.  **Define Decision Variables:**
    -   `x[i]` = Integer quantity of item_ref i to place (must be 0 if not authorized; otherwise, between minimum_lot and maximum_order). Type: GRB.INTEGER.
    -   `y[i]` = 1 if item_ref i is selected (i.e., x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `z[c]` = 1 if any item in category c is selected (i.e., sum of x[i] for items in c > 0), 0 otherwise. Type: GRB.BINARY.
    -   `b[bundle]` = 1 if both items in bundle are selected, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients:
        - unit_benefit_cents (per unit, from item tables)
        - item_fee_cents (fixed, per item if selected, from item tables)
        - activation_fee_cents (fixed, per category if any item in category is selected, from category table)
        - bonus_cents (per bundle, from bundle table)
    -   Constraint coefficients:
        - usage amount per item per resource (from usage tables)
        - capacity_ledger (sum of opening and reservation per resource/area)
        - minimum_lot, maximum_order (from item tables)
        - minimum_quantity, maximum_quantity per category (from category table)
    -   Logical constraints:
        - authorized (from item tables)
        - incompatible pairs (from incompatible table)
        - requires pairs (from requires table)
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    - Sum over all items: (unit_benefit_cents[i] * x[i]) 
    - Minus sum over all items: (item_fee_cents[i] * y[i]) [fixed fee if item selected]
    - Minus sum over all categories: (activation_fee_cents[c] * z[c]) [fixed fee if any item in category selected]
    - Plus sum over all bundles: (bonus_cents[bundle] * b[bundle]) [bonus if both items in bundle selected]
7.  **Formulate Constraints:**
    -   **Item Authorization:** For each item i, if authorized == 0, x[i] = 0 and y[i] = 0.
    -   **Item Quantity Bounds:** For each authorized item i, x[i] = 0 or minimum_lot[i] ≤ x[i] ≤ maximum_order[i].
    -   **Item Activation Linking:** For each item i, y[i] = 1 if x[i] > 0, else y[i] = 0 (enforced via x[i] ≤ maximum_order[i] * y[i] and x[i] ≥ minimum_lot[i] * y[i]).
    -   **Resource/Area Capacity:** For each resource/area r, sum over all items assigned to r of (usage_amount[i, r] * x[i]) ≤ total available capacity for r (sum of opening and reservation in capacity_ledger for r).
    -   **Category Quantity Bounds:** For each category c, sum over all items in c of x[i] ≥ minimum_quantity[c] * z[c] and ≤ maximum_quantity[c] * z[c]; also, sum over all items in c of x[i] ≥ minimum_quantity[c] if z[c] = 1.
    -   **Category Activation Linking:** For each category c, z[c] = 1 if any x[i] > 0 for i in c, else z[c] = 0.
    -   **Incompatibility:** For each incompatible pair (i, j), y[i] + y[j] ≤ 1 (cannot select both).
    -   **Requires:** For each requires pair (i, prereq), y[i] ≤ y[prereq] (cannot select i unless prereq is also selected).
    -   **Bundle Bonuses:** For each bundle (i, j), b[bundle] ≤ y[i], b[bundle] ≤ y[j], b[bundle] ≥ y[i] + y[j] - 1 (b[bundle] = 1 iff both y[i] = y[j] = 1).
    -   **Integrality:** All x[i] are integer, y[i], z[c], b[bundle] are binary.
[Abstract Model Plan END]