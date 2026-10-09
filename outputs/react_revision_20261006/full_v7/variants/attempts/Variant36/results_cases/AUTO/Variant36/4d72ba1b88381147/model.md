[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer quantities of authorized air-conditioner models to assign to specific storage areas at FC_EAST_HVAC, maximizing net value (total benefit plus bonuses, minus all fixed and variable fees), subject to area volume limits, category quantity bounds, incompatibility and prerequisite rules, and bundle bonuses.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation) costs, resource constraints, logical (incompatibility/prerequisite) constraints, and bundle bonuses.
3.  **Define Index Sets:** The primary indices are:
    - Items (`i`): Each authorized item_ref from the item tables (all rows with authorized=1).
    - Categories (`g`): Each category from the category table.
    - Resources/Areas (`r`): Each resource/location_id from the capacity_ledger and usage tables (e.g., AREA_A, AREA_B, AREA_C).
    - Bundles (`b`): Each bundle row (item_a, item_b) from the bundle table.
    - Incompatibility pairs and prerequisite pairs from their respective tables.
4.  **Define Decision Variables:**
    -   `x[i]` = Integer quantity of item `i` to assign (must be 0 or between minimum_lot and maximum_order for authorized items; 0 for unauthorized). Type: GRB.INTEGER.
    -   `y[i]` = 1 if item `i` is selected (i.e., x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `z[g]` = 1 if any item in category `g` is selected (i.e., sum of x[i] for items in `g` > 0), 0 otherwise. Type: GRB.BINARY.
    -   `w[b]` = 1 if both items in bundle `b` are selected (i.e., y[item_a] = y[item_b] = 1), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients:
        -   `unit_benefit_cents` (from item tables): per-unit benefit for each item.
        -   `item_fee_cents` (from item tables): fixed fee per item if any units are selected.
        -   `activation_fee_cents` (from category table): fixed fee per category if any item in the category is selected.
        -   `bonus_cents` (from bundle table): bonus if both items in the bundle are selected.
    -   Constraint coefficients:
        -   `usage` (from usage tables): amount of resource used per unit of each item.
        -   `capacity_ledger` (from capacity_ledger table): total available capacity per resource (sum opening + reservations).
        -   `minimum_lot`, `maximum_order` (from item tables): lower and upper bounds for each item’s quantity.
        -   `minimum_quantity`, `maximum_quantity` (from category table): lower and upper bounds for total quantity per category.
        -   Incompatibility pairs (from incompatible table): pairs of items that cannot both be selected.
        -   Prerequisite pairs (from requires table): if item_ref is selected, prerequisite_ref must also be selected.
        -   Authorization (from item tables): only items with authorized=1 may be selected (others must have x[i]=0).
6.  **Formulate Objective:** Maximize total net benefit in cents:
        -   Sum over all items: (unit_benefit_cents[i] * x[i]) 
        -   Minus sum over all items: (item_fee_cents[i] * y[i]) [fixed fee if any units of item i are selected]
        -   Minus sum over all categories: (activation_fee_cents[g] * z[g]) [fixed fee if any item in category g is selected]
        -   Plus sum over all bundles: (bonus_cents[b] * w[b]) [bonus if both items in bundle b are selected]
7.  **Formulate Constraints:**
    -   **Item Authorization:** For each item i, if authorized=0, enforce x[i]=0 and y[i]=0.
    -   **Item Quantity Bounds:** For each authorized item i, x[i] = 0 or minimum_lot[i] ≤ x[i] ≤ maximum_order[i].
    -   **Item Activation:** For each item i, y[i]=1 if x[i]>0, y[i]=0 if x[i]=0 (enforce via x[i] ≤ maximum_order[i]*y[i] and x[i] ≥ minimum_lot[i]*y[i]).
    -   **Resource (Area) Capacity:** For each resource r, sum over all items assigned to r of (usage[r][i] * x[i]) ≤ total available capacity for r (sum of opening + reservations from capacity_ledger).
    -   **Category Quantity Bounds:** For each category g, sum over all items in g of x[i] must satisfy minimum_quantity[g] ≤ sum ≤ maximum_quantity[g].
    -   **Category Activation:** For each category g, z[g]=1 if any x[i]>0 for i in g, z[g]=0 otherwise (enforce via sum over i in g of x[i] ≤ bigM*z[g], and sum over i in g of x[i] ≥ minimum_lot_min*g*z[g] if needed).
    -   **Incompatibility:** For each incompatible pair (i,j), y[i] + y[j] ≤ 1.
    -   **Prerequisite:** For each requires pair (i,prereq), y[i] ≤ y[prereq].
    -   **Bundle Bonuses:** For each bundle (item_a, item_b), w[b] ≤ y[item_a], w[b] ≤ y[item_b], w[b] ≥ y[item_a] + y[item_b] - 1 (w[b]=1 iff both y[item_a]=y[item_b]=1).
    -   **Integrality:** All x[i] are integer, all y[i], z[g], w[b] are binary.
[Abstract Model Plan END]