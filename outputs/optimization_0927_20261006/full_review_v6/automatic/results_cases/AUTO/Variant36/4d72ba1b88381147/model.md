[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer quantities of authorized air-conditioner models to assign to each storage area at FC_EAST_HVAC, maximizing net value (total benefit plus bonuses minus all fixed and activation fees), subject to per-area volume limits, item/category bounds, incompatibility and prerequisite rules, and bundle bonuses.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge, group activation, and logical (incompatibility, prerequisite, bundle) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items (`i`): All item rows from the union of item tables (from both batch_06/export_06.csv and batch_01/export_07.csv).
    - Categories (`g`): All category rows from the category table (batch_03/export_03.csv).
    - Resources/Areas (`r`): All resources from the capacity_ledger and usage tables (AREA_A, AREA_B, AREA_C).
    - Bundles (`(i,j)`): All bundle pairs from the bundle table (batch_01/export_01.csv).
    - Incompatible pairs (`(i,j)`): All pairs from the incompatible table (batch_05/export_05.csv).
    - Requires pairs (`(i,k)`): All pairs from the requires table (batch_03/export_09.csv).
4.  **Define Decision Variables:**
    -   `x[i]` = Integer quantity of item `i` to assign (0 if not selected, else in [minimum_lot, maximum_order] for authorized items; 0 for unauthorized).
    -   `y[i]` = Binary variable: 1 if item `i` is selected (i.e., x[i] > 0), 0 otherwise.
    -   `z[g]` = Binary variable: 1 if any item in category `g` is selected, 0 otherwise.
    -   `b[i,j]` = Binary variable: 1 if both items `i` and `j` in a bundle are selected, 0 otherwise.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients:
        -   `unit_benefit_cents[i]` (from item tables): benefit per unit of item `i`.
        -   `item_fee_cents[i]` (from item tables): fixed fee per item `i` if selected.
        -   `activation_fee_cents[g]` (from category table): fixed fee per category `g` if any item in `g` is selected.
        -   `bonus_cents[i,j]` (from bundle table): bonus if both items `i` and `j` are selected.
    -   Constraint coefficients:
        -   `usage[i,r]` (from usage tables): resource usage per unit of item `i` in resource/area `r`.
        -   `capacity[r]` (from capacity_ledger): total available capacity for resource/area `r` (sum of all entries for each resource).
        -   `minimum_lot[i]`, `maximum_order[i]` (from item tables): lower and upper bounds for item `i`.
        -   `authorized[i]` (from item tables): 1 if item `i` is eligible, 0 otherwise.
        -   `category[i]` (from item tables): category assignment for item `i`.
        -   `minimum_quantity[g]`, `maximum_quantity[g]` (from category table): lower and upper bounds for total quantity in category `g`.
    -   Logical constraints:
        -   Incompatible pairs: from incompatible table.
        -   Requires pairs: from requires table.
6.  **Formulate Objective:** Maximize total net benefit in cents:
        sum over items [unit_benefit_cents[i] * x[i] - item_fee_cents[i] * y[i]]
      plus
        sum over bundles [bonus_cents[i,j] * b[i,j]]
      minus
        sum over categories [activation_fee_cents[g] * z[g]]
7.  **Formulate Constraints:**
    -   **Resource/Area Capacity:** For each resource/area `r`, sum over items assigned to `r` of (usage[i,r] * x[i]) ≤ total capacity[r] (sum of all capacity_ledger entries for `r`).
    -   **Item Selection and Bounds:** For each item `i`:
            - x[i] = 0 if authorized[i] = 0.
            - If authorized[i] = 1: x[i] ∈ {0} ∪ [minimum_lot[i], maximum_order[i]] (enforced via y[i]: minimum_lot[i] * y[i] ≤ x[i] ≤ maximum_order[i] * y[i]; x[i] = 0 iff y[i] = 0).
    -   **Category Quantity Bounds:** For each category `g`, sum over items in `g` of x[i] ∈ [minimum_quantity[g], maximum_quantity[g]] (unconditional, applies even if no item is selected).
    -   **Category Activation:** For each category `g`, z[g] = 1 iff any y[i] = 1 for i in g; z[g] = 0 otherwise.
    -   **Incompatibility:** For each incompatible pair (i,j), y[i] + y[j] ≤ 1.
    -   **Requires:** For each requires pair (i,k), y[i] ≤ y[k] (if i is selected, k must be selected).
    -   **Bundle Bonuses:** For each bundle (i,j), b[i,j] = 1 iff y[i] = 1 and y[j] = 1; b[i,j] ≤ y[i], b[i,j] ≤ y[j], b[i,j] ≥ y[i] + y[j] - 1.
    -   **Variable Domains:** x[i] ∈ ℤ≥0, y[i], z[g], b[i,j] ∈ {0,1}.
[Abstract Model Plan END]