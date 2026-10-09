[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer quantities of authorized air-conditioner models to place in each storage area at FC_EAST_HVAC, maximizing net value (in USD cents). The plan must respect area-specific volume limits, item and category constraints, incompatibility and prerequisite rules, and account for fixed and bonus fees.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge, set-packing, and logical (compatibility/prerequisite) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items (`item_ref`): Each air-conditioner model/option.
    - Categories (`category`): Each item belongs to a category.
    - Areas/Resources (`location_id`/`resource`): Each storage area with its own capacity.
    - Bundles: Pairs of items eligible for a bonus if both are selected.
    - Incompatibility pairs: Pairs of items that cannot be selected together.
    - Prerequisite pairs: Pairs where one item requires another to be selected.
4.  **Define Decision Variables:**
    -   `x[i]` = Integer quantity of item `i` to place (for each authorized item). Type: GRB.INTEGER, with bounds [0, maximum_order[i]] and 0 unless authorized.
    -   `y[i]` = Binary variable: 1 if any quantity of item `i` is selected (i.e., x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `z[c]` = Binary variable: 1 if any item in category `c` is selected (i.e., category is "active"), 0 otherwise. Type: GRB.BINARY.
    -   `b[bundle]` = Binary variable: 1 if both items in bundle are selected (i.e., both y[i_a] and y[i_b] = 1), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients:
        -   `unit_benefit_cents` (from item tables): per-unit benefit for each item.
        -   `item_fee_cents` (from item tables): fixed fee per item if any quantity is selected.
        -   `activation_fee_cents` (from category table): fixed fee per category if any item in the category is selected.
        -   `bonus_cents` (from bundle table): bonus if both items in a bundle are selected.
    -   Constraint coefficients:
        -   `usage` (from usage tables): per-unit resource usage for each item in each area.
        -   `capacity_ledger` (from capacity_ledger table): net available capacity per area/resource.
        -   `minimum_lot`, `maximum_order` (from item tables): lower/upper bounds for each item.
        -   `minimum_quantity`, `maximum_quantity` (from category table): lower/upper bounds for total quantity per category.
        -   `authorized` (from item tables): only items with authorized=1 can be selected.
        -   Incompatibility and prerequisite pairs (from incompatible and requires tables).
    -   Constraint RHS:
        -   Net capacity per area/resource (sum of opening and reservation entries).
        -   Category quantity limits.
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    -   Sum over all items: (unit_benefit_cents[i] * x[i]) 
    -   Minus sum over all items: (item_fee_cents[i] * y[i]) [fixed fee if any of item i is selected]
    -   Minus sum over all categories: (activation_fee_cents[c] * z[c]) [fixed fee if any item in category c is selected]
    -   Plus sum over all bundles: (bonus_cents[bundle] * b[bundle]) [bonus if both items in bundle are selected]
7.  **Formulate Constraints:**
    -   **Item selection and bounds:**
        -   For each item i: x[i] = 0 if authorized[i] = 0; else x[i] ∈ {0} ∪ [minimum_lot[i], maximum_order[i]].
        -   For each item i: y[i] = 1 if x[i] ≥ minimum_lot[i]; y[i] = 0 if x[i] = 0.
    -   **Resource/Area capacity:**
        -   For each area/resource r: sum over items assigned to r of (usage[r][i] * x[i]) ≤ net capacity[r] (from capacity_ledger).
    -   **Category quantity and activation:**
        -   For each category c: sum over items in c of x[i] ∈ [minimum_quantity[c], maximum_quantity[c]] if z[c]=1; else = 0.
        -   For each category c: z[c] = 1 if any x[i] > 0 for i in c; z[c] = 0 otherwise.
    -   **Incompatibility:**
        -   For each incompatible pair (i, j): y[i] + y[j] ≤ 1.
    -   **Prerequisite:**
        -   For each requires pair (i, j): y[i] ≤ y[j] (if i requires j).
    -   **Bundle bonuses:**
        -   For each bundle (i, j): b[bundle] ≤ y[i], b[bundle] ≤ y[j], b[bundle] ≥ y[i] + y[j] - 1.
    -   **Integrality and binary:**
        -   x[i] ∈ {0} ∪ [minimum_lot[i], maximum_order[i]] (integer).
        -   y[i], z[c], b[bundle] ∈ {0,1}.
[Abstract Model Plan END]