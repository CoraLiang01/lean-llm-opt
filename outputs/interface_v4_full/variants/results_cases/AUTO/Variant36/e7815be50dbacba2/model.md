[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer quantities of air-conditioner models (item_ref) to assign to storage areas (location_id) at FC_EAST_HVAC, maximizing net value (in USD cents). The plan must respect area-specific volume limits, item and category constraints, incompatibilities, prerequisites, and bundle bonuses, using only authorized options and adhering to minimum/maximum order sizes and category activation fees.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation) and logical (incompatibility, prerequisite, bundle) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items: All item_ref rows from the supplied item tables (export_06.csv and export_07.csv).
    - Categories: All category rows from the category table (export_03.csv).
    - Areas/Resources: All resource/location_id values from the capacity_ledger and usage tables.
    - Bundles: All bundle rows from the bundle table (export_01.csv).
    - Incompatibilities: All incompatible pairs from the incompatible table (export_05.csv).
    - Prerequisites: All requires pairs from the requires table (export_09.csv).
4.  **Define Decision Variables:**
    -   `x[i]` = Integer quantity of item_ref i to assign (must be 0 if unauthorized; otherwise, 0 or an integer between minimum_lot and maximum_order for i). Type: GRB.INTEGER.
    -   `y[i]` = 1 if any positive quantity of item_ref i is chosen (i.e., x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `z[c]` = 1 if any item in category c is selected (i.e., sum of x[i] for items in c > 0), 0 otherwise. Type: GRB.BINARY.
    -   `b[bundle]` = 1 if both items in bundle are selected (i.e., y[item_a] = 1 and y[item_b] = 1), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients:
        -   unit_benefit_cents (from item tables) for each item_ref.
        -   item_fee_cents (from item tables) for each item_ref (fixed cost if any units chosen).
        -   activation_fee_cents (from category table) for each category (fixed cost if any item in category is chosen).
        -   bonus_cents (from bundle table) for each bundle (bonus if both items are chosen).
    -   Constraint coefficients:
        -   usage amount per item_ref and resource (from usage tables export_10.csv and export_11.csv).
        -   capacity_ledger amounts per resource (from export_02.csv), summed by resource for total available capacity.
        -   minimum_lot and maximum_order per item_ref (from item tables).
        -   minimum_quantity and maximum_quantity per category (from category table).
        -   authorized flag per item_ref (from item tables).
        -   incompatible pairs (from incompatible table).
        -   requires pairs (from requires table).
    -   Constraint RHS:
        -   Total available capacity per resource (sum of capacity_ledger entries for each resource).
        -   Category quantity limits (minimum_quantity and maximum_quantity).
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    -   Sum over all items: (unit_benefit_cents[i] * x[i]) 
    -   Minus sum over all items: (item_fee_cents[i] * y[i]) [fixed fee if any units of i chosen]
    -   Minus sum over all categories: (activation_fee_cents[c] * z[c]) [fixed fee if any item in c chosen]
    -   Plus sum over all bundles: (bonus_cents[bundle] * b[bundle]) [bonus if both items in bundle are chosen]
7.  **Formulate Constraints:**
    -   **Item Authorization and Order Size:**
        -   For each item_ref i: If authorized[i] = 0, then x[i] = 0.
        -   For each authorized item_ref i: x[i] = 0 or x[i] ∈ [minimum_lot[i], maximum_order[i]] (i.e., x[i] = 0 or x[i] ≥ minimum_lot[i] and x[i] ≤ maximum_order[i]).
        -   For each item_ref i: y[i] = 1 if x[i] > 0, else y[i] = 0 (enforced via x[i] ≤ maximum_order[i] * y[i], x[i] ≥ minimum_lot[i] * y[i]).
    -   **Resource (Area) Capacity:**
        -   For each resource/area r: sum over all items assigned to r of (usage_amount[i, r] * x[i]) ≤ total available capacity for r (sum of capacity_ledger amounts for r).
    -   **Category Quantity and Activation:**
        -   For each category c: sum over all items in c of x[i] ≥ minimum_quantity[c] * z[c] and ≤ maximum_quantity[c] * z[c].
        -   For each category c: z[c] = 1 if any x[i] > 0 for i in c, else 0.
    -   **Incompatibility:**
        -   For each incompatible pair (i, j): y[i] + y[j] ≤ 1 (cannot select both).
    -   **Prerequisite (Requires):**
        -   For each requires pair (i, prereq): y[i] ≤ y[prereq] (if i is selected, prereq must also be selected).
    -   **Bundle Bonuses:**
        -   For each bundle (item_a, item_b): b[bundle] ≤ y[item_a], b[bundle] ≤ y[item_b], b[bundle] ≥ y[item_a] + y[item_b] - 1 (b[bundle] = 1 iff both items are selected).
    -   **Variable Domains:**
        -   x[i] ∈ {0} ∪ [minimum_lot[i], maximum_order[i]] for authorized items; x[i] = 0 for unauthorized.
        -   y[i], z[c], b[bundle] ∈ {0, 1}.
[Abstract Model Plan END]