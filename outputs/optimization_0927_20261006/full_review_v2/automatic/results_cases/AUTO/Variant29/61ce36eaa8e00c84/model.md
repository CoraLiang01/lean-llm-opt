[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer order quantities for each authorized bread option to maximize net benefit (in USD cents), considering per-unit benefits, fixed item and category activation fees, bundle bonuses, resource capacities (labor, space, power), category quantity bounds, item incompatibilities, and prerequisite requirements.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge, resource allocation, and logical (incompatibility, prerequisite, bundle) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items (`i`): All rows from the 'item' table in batch_03/export_03.csv with `authorized=1`.
    - Categories (`c`): All categories from the 'category' table in batch_05/export_05.csv.
    - Resources (`r`): All resources from the 'usage' and 'capacity_ledger' tables (labor, space, power).
    - Incompatible pairs (`(i,j)`): All pairs from the 'incompatible' table in batch_06/export_06.csv.
    - Prerequisite pairs (`(i,p)`): All pairs from the 'requires' table in batch_01/export_01.csv.
    - Bundles (`(i,j)`): All pairs from the 'bundle' table in batch_02/export_08.csv.
4.  **Define Decision Variables:**
    -   `x[i]` = Integer quantity ordered of item `i` (authorized items only). Type: GRB.INTEGER, domain: {0} ∪ [minimum_lot[i], maximum_order[i]].
    -   `y[i]` = 1 if item `i` is ordered in any positive quantity, 0 otherwise. Type: GRB.BINARY.
    -   `z[c]` = 1 if any item in category `c` is ordered, 0 otherwise. Type: GRB.BINARY.
    -   `b[i,j]` = 1 if both items `i` and `j` in a bundle are ordered in positive quantities, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients:
        -   `unit_benefit_cents[i]` (from 'item' table): per-unit benefit for item `i`.
        -   `item_fee_cents[i]` (from 'item' table): fixed fee if any of item `i` is ordered.
        -   `activation_fee_cents[c]` (from 'category' table): fixed fee if any item in category `c` is ordered.
        -   `bonus_cents[i,j]` (from 'bundle' table): bonus if both items in bundle are ordered.
    -   Constraint coefficients:
        -   `minimum_lot[i]`, `maximum_order[i]` (from 'item' table): unconditional lower/upper bounds for item quantities.
        -   `minimum_quantity[c]`, `maximum_quantity[c]` (from 'category' table): unconditional lower/upper bounds for total quantity in each category.
        -   `amount[i,r]` and `unit[i,r]` (from 'usage' table): per-unit resource usage for item `i` and resource `r`.
        -   `amount[r]` and `unit[r]` (from 'capacity_ledger' table): signed sum of resource capacity for resource `r`.
    -   Logical constraints:
        -   Incompatible pairs (`item_a`, `item_b` from 'incompatible' table).
        -   Prerequisite pairs (`item_ref`, `prerequisite_ref` from 'requires' table).
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    -   Sum over all items: (unit_benefit_cents[i] * x[i]) 
    -   Minus sum over all items: (item_fee_cents[i] * y[i])
    -   Minus sum over all categories: (activation_fee_cents[c] * z[c])
    -   Plus sum over all bundles: (bonus_cents[i,j] * b[i,j])
7.  **Formulate Constraints:**
    -   **Item Quantity Bounds:** For each authorized item `i`, enforce unconditional bounds: x[i] = 0 or minimum_lot[i] ≤ x[i] ≤ maximum_order[i].
    -   **Item Activation Linking:** For each item `i`, y[i] = 1 if x[i] ≥ minimum_lot[i], y[i] = 0 if x[i] = 0. (Link x[i] and y[i].)
    -   **Category Activation Linking:** For each category `c`, z[c] = 1 if any x[i] > 0 for items in category `c`, else z[c] = 0.
    -   **Category Quantity Bounds:** For each category `c`, sum over items in `c`: minimum_quantity[c] ≤ Σ x[i] ≤ maximum_quantity[c] (unconditional, even if z[c]=0).
    -   **Resource Capacity:** For each resource `r`, sum over items: Σ (amount[i,r] * x[i] * unit_conversion_factor[r]) ≤ total available capacity for resource `r` (convert all units to base units: ml, minutes, wh).
    -   **Incompatibility:** For each incompatible pair (i,j), y[i] + y[j] ≤ 1 (cannot order both).
    -   **Prerequisite:** For each (i,p) in 'requires', y[i] ≤ y[p] (cannot order i unless prerequisite p is also ordered).
    -   **Bundle Bonus Linking:** For each bundle (i,j), b[i,j] ≤ y[i], b[i,j] ≤ y[j], b[i,j] ≥ y[i] + y[j] - 1 (b[i,j]=1 iff both y[i]=y[j]=1).
    -   **Authorization:** Only items with authorized=1 are included in the model; unauthorized items cannot be ordered or trigger bonuses.
[Abstract Model Plan END]