[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer quantities of authorized air-conditioner models to assign to storage areas at FC_EAST_HVAC, maximizing net value (in USD cents). The plan must respect per-area volume limits, per-category quantity bounds, item and category activation fees, incompatibility and prerequisite (requires) relationships, and bundle bonuses. Only authorized options may be chosen, and each item must be ordered in integer multiples within its allowed lot/order range.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation) costs, resource constraints, logical (incompatibility and requires) constraints, and bundle bonuses.
3.  **Define Index Sets:** The primary indices are:
    - Items (`I`): All item_ref rows from the union of the two item tables (export_06.csv and export_07.csv), filtered to authorized=1.
    - Categories (`C`): All category rows from export_03.csv.
    - Resources/Areas (`R`): All resource/location_id values from capacity_ledger and usage tables (AREA_A, AREA_B, AREA_C).
    - Bundles (`B`): All bundle rows from export_01.csv.
    - Incompatibilities (`Inc`): All incompatible item pairs from export_05.csv.
    - Requires (`Req`): All requires pairs from export_09.csv.
4.  **Define Decision Variables:**
    -   `x[i]` = Integer quantity of item `i` selected (0 if not chosen). Type: GRB.INTEGER.
    -   `y[i]` = 1 if item `i` is selected (i.e., x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `z[c]` = 1 if any item in category `c` is selected (category is "active"), 0 otherwise. Type: GRB.BINARY.
    -   `b[bundle]` = 1 if both items in bundle `b` are selected (for bundle bonus), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients:
        -   `unit_benefit_cents[i]` (from item tables): benefit per unit of item `i`.
        -   `item_fee_cents[i]` (from item tables): fixed fee if any of item `i` is selected.
        -   `activation_fee_cents[c]` (from category table): fixed fee if any item in category `c` is selected.
        -   `bonus_cents[b]` (from bundle table): bonus if both items in bundle `b` are selected.
    -   Constraints:
        -   `minimum_lot[i]`, `maximum_order[i]` (from item tables): lower/upper bounds for x[i] if selected.
        -   `usage[i,r]` (from usage tables): amount of resource `r` used per unit of item `i`.
        -   `capacity[r]` (from capacity_ledger): sum of opening and reservation for each resource/area.
        -   `minimum_quantity[c]`, `maximum_quantity[c]` (from category table): lower/upper bounds for total quantity in category `c`.
        -   Incompatibility pairs (from incompatible table).
        -   Requires pairs (from requires table).
        -   Authorization: only items with authorized=1 may be selected (x[i]=0 for others).
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    -   Sum over all items: (unit_benefit_cents[i] * x[i]) 
    -   Minus sum over all items: (item_fee_cents[i] * y[i]) [fixed fee if any of item i is chosen]
    -   Minus sum over all categories: (activation_fee_cents[c] * z[c]) [fixed fee if any item in category c is chosen]
    -   Plus sum over all bundles: (bonus_cents[b] * b[b]) [bonus if both items in bundle b are chosen]
7.  **Formulate Constraints:**
    -   **Item selection and bounds:**
        -   For each item i: x[i] = 0 if authorized[i] = 0.
        -   For each item i: x[i] = 0 or minimum_lot[i] ≤ x[i] ≤ maximum_order[i] (if selected).
        -   For each item i: y[i] = 1 if x[i] ≥ minimum_lot[i], 0 otherwise.
    -   **Resource/Area capacity:**
        -   For each resource/area r: sum over items assigned to r of (usage[i,r] * x[i]) ≤ capacity[r].
    -   **Category quantity and activation:**
        -   For each category c: minimum_quantity[c] ≤ sum over items in c of x[i] ≤ maximum_quantity[c].
        -   For each category c: z[c] = 1 if any x[i] > 0 for i in c, 0 otherwise.
    -   **Incompatibility:**
        -   For each incompatible pair (i,j): y[i] + y[j] ≤ 1.
    -   **Requires:**
        -   For each requires pair (i,prereq): y[i] ≤ y[prereq].
    -   **Bundle bonuses:**
        -   For each bundle (i,j): b[bundle] ≤ y[i], b[bundle] ≤ y[j], b[bundle] ≥ y[i] + y[j] - 1.
    -   **Variable domains:**
        -   x[i] ∈ {0} ∪ [minimum_lot[i], maximum_order[i]] (integer), y[i], z[c], b[bundle] ∈ {0,1}.
[Abstract Model Plan END]