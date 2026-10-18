[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer quantities of authorized air-conditioner models to assign to each storage area at FC_EAST_HVAC, maximizing total net value (in USD cents). The plan must respect per-area volume limits, item and category quantity bounds, fixed and activation fees, incompatibility and prerequisite requirements, and bundle bonuses, using only the supplied item rows.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge, group activation, logical (incompatibility, requires), and resource constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items (`i`): All item rows from the union of the two 'item' tables (from both batch_06/export_06.csv and batch_01/export_07.csv).
    - Categories (`g`): All category rows from the 'category' table.
    - Resources/Areas (`r`): All resources from the 'capacity_ledger' and 'usage' tables (e.g., AREA_A, AREA_B, AREA_C).
    - Bundles (`b`): All bundle rows from the 'bundle' table.
    - Incompatibility pairs (`(i,j)`): All pairs from the 'incompatible' table.
    - Requires pairs (`(i,j)`): All pairs from the 'requires' table.
4.  **Define Decision Variables:**
    -   `x[i]` = Integer quantity of item `i` to assign (0 or in [minimum_lot[i], maximum_order[i]] if authorized; 0 if unauthorized). Type: GRB.INTEGER.
    -   `y[i]` = 1 if item `i` is selected (i.e., x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `z[g]` = 1 if any item in category `g` is selected (i.e., at least one y[i] for i in g is 1), 0 otherwise. Type: GRB.BINARY.
    -   `w[b]` = 1 if both items in bundle `b` are selected (i.e., y[item_a[b]] = y[item_b[b]] = 1), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients:
        -   `unit_benefit_cents[i]` (from 'item' tables): benefit per unit of item `i`.
        -   `item_fee_cents[i]` (from 'item' tables): fixed fee per item `i` if selected.
        -   `activation_fee_cents[g]` (from 'category' table): fixed fee per category `g` if any item in `g` is selected.
        -   `bonus_cents[b]` (from 'bundle' table): bonus if both items in bundle `b` are selected.
    -   Constraint coefficients:
        -   `usage[i,r]` (from 'usage' tables): amount of resource `r` used per unit of item `i`.
    -   Constraint RHS:
        -   `capacity[r]` (sum of 'amount' for each resource `r` in 'capacity_ledger' table): total available capacity per resource/area.
        -   `minimum_lot[i]`, `maximum_order[i]` (from 'item' tables): per-item quantity bounds.
        -   `minimum_quantity[g]`, `maximum_quantity[g]` (from 'category' table): per-category aggregate quantity bounds.
        -   `authorized[i]` (from 'item' tables): eligibility flag for item `i`.
        -   Incompatibility and requires pairs (from 'incompatible' and 'requires' tables).
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    -   Sum over all items: (unit_benefit_cents[i] * x[i]) minus (item_fee_cents[i] * y[i])
    -   Minus sum over all categories: (activation_fee_cents[g] * z[g])
    -   Plus sum over all bundles: (bonus_cents[b] * w[b])
7.  **Formulate Constraints:**
    -   **Resource/Area Capacity:** For each resource/area `r`, sum over all items assigned to `r` of (usage[i,r] * x[i]) ≤ capacity[r]. No borrowing between areas.
    -   **Item Authorization and Quantity Bounds:** For each item `i`, x[i] = 0 if authorized[i] = 0; else, x[i] ∈ {0} ∪ [minimum_lot[i], maximum_order[i]] (enforced via y[i] and bounds: minimum_lot[i] * y[i] ≤ x[i] ≤ maximum_order[i] * y[i]).
    -   **Item Selection Flag:** For each item `i`, y[i] = 1 iff x[i] > 0; y[i] ∈ {0,1}.
    -   **Category Quantity Bounds:** For each category `g`, sum over items in `g` of x[i] ∈ [minimum_quantity[g], maximum_quantity[g]] (unconditional, even if no item is selected).
    -   **Category Activation Flag:** For each category `g`, z[g] = 1 iff any y[i] for i in g is 1; for all i in g, y[i] ≤ z[g]; z[g] ≤ sum(y[i] for i in g).
    -   **Incompatibility:** For each incompatible pair (i,j), y[i] + y[j] ≤ 1.
    -   **Requires:** For each requires pair (i,j), y[i] ≤ y[j] (i requires j).
    -   **Bundle Bonus:** For each bundle b = (i,j), w[b] = 1 iff y[i] = y[j] = 1; w[b] ≤ y[i], w[b] ≤ y[j], w[b] ≥ y[i] + y[j] - 1.
    -   **Variable Domains:** All x[i] are integer, all y[i], z[g], w[b] are binary.
[Abstract Model Plan END]