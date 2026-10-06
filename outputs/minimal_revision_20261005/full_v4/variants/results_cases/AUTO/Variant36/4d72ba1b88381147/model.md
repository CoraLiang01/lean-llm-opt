[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer quantities of authorized air-conditioner models to place in each storage area at FC_EAST_HVAC, maximizing net value (in USD cents). The plan must respect area volume limits, per-category quantity bounds, item and category activation fees, incompatibility and prerequisite (requires) relationships, and bundle bonuses. Only authorized options may be selected, and each area’s capacity is independent.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation) costs, resource constraints, logical (incompatibility and requires) constraints, and bundle bonuses.
3.  **Define Index Sets:** The primary indices are:
    - Items (`item_ref`): Each authorized air-conditioner model/option.
    - Areas (`location_id`): Storage areas (e.g., AREA_A, AREA_B, AREA_C).
    - Categories (`category`): Each item belongs to a category (e.g., G0, G1, G2, G3).
    - Resources: Each area’s volume/capacity constraint.
    - Bundles: Pairs of items eligible for a bonus if both are selected.
    - Incompatibilities: Pairs of items that cannot be selected together.
    - Requires: Pairs where one item requires another to be selected.
4.  **Define Decision Variables:**
    -   `x[i]` = Integer quantity of item `i` to place (for each authorized item_ref). Type: GRB.INTEGER. Must be 0 or between `minimum_lot` and `maximum_order` for that item.
    -   `y[i]` = Binary variable: 1 if item `i` is selected (i.e., `x[i] > 0`), 0 otherwise. Type: GRB.BINARY.
    -   `z[c]` = Binary variable: 1 if any item in category `c` is selected (i.e., category is active), 0 otherwise. Type: GRB.BINARY.
    -   `b[bundle]` = Binary variable: 1 if both items in bundle are selected, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients:
        -   `unit_benefit_cents` (from item tables): Earned per unit of item placed.
        -   `item_fee_cents` (from item tables): Paid once per item if any units are selected.
        -   `activation_fee_cents` (from category table): Paid once per category if any item in that category is selected.
        -   `bonus_cents` (from bundle table): Earned once per bundle if both items are selected.
    -   Constraint coefficients:
        -   `usage` (from usage tables): Amount of resource (e.g., volume in ml) used per unit of item in each area.
        -   `capacity_ledger` (from capacity_ledger table): Net available capacity per area/resource (sum of opening and reservation).
        -   `minimum_lot`, `maximum_order` (from item tables): Lower and upper bounds for each item’s quantity.
        -   `minimum_quantity`, `maximum_quantity` (from category table): Lower and upper bounds for total quantity per category.
        -   `authorized` (from item tables): Only items with authorized=1 may be selected.
        -   Incompatibilities and requires (from respective tables): Logical relationships between items.
    -   Sets for bundles, incompatibilities, and requires are defined by their respective tables.
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    -   Sum over all items: (unit_benefit_cents[i] * x[i]) 
    -   Minus sum over all items: (item_fee_cents[i] * y[i]) [charged once per item if selected]
    -   Minus sum over all categories: (activation_fee_cents[c] * z[c]) [charged once per category if any item in c is selected]
    -   Plus sum over all bundles: (bonus_cents[bundle] * b[bundle]) [awarded if both items in bundle are selected]
7.  **Formulate Constraints:**
    -   **Item Authorization:** For each item, if authorized=0, force x[i]=0 and y[i]=0.
    -   **Item Quantity Bounds:** For each authorized item, x[i]=0 or x[i] in [minimum_lot, maximum_order] (enforced via y[i]: x[i] ≥ minimum_lot*y[i], x[i] ≤ maximum_order*y[i]).
    -   **Item Activation:** For each item, y[i]=1 if x[i]>0, y[i]=0 if x[i]=0.
    -   **Area Resource Constraints:** For each area/resource, sum over all items assigned to that area of (usage per unit * x[i]) ≤ net available capacity (sum of opening and reservation in capacity_ledger for that area).
    -   **Category Quantity Bounds:** For each category, sum of x[i] over all items in that category ∈ [minimum_quantity, maximum_quantity] (from category table).
    -   **Category Activation:** For each category, z[c]=1 if any x[i]>0 for items in c, z[c]=0 otherwise.
    -   **Incompatibility Constraints:** For each incompatible pair (i,j), y[i] + y[j] ≤ 1 (cannot both be selected).
    -   **Requires Constraints:** For each requires pair (i,prereq), y[i] ≤ y[prereq] (if i is selected, prereq must also be selected).
    -   **Bundle Constraints:** For each bundle (i,j), b[bundle] ≤ y[i], b[bundle] ≤ y[j], b[bundle] ≥ y[i] + y[j] - 1 (b[bundle]=1 iff both y[i]=1 and y[j]=1).
    -   **Non-negativity and Integrality:** All x[i] ≥ 0 and integer; all y[i], z[c], b[bundle] ∈ {0,1}.
[Abstract Model Plan END]