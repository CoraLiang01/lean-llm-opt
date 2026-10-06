[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer replenishment quantities for each authorized option (item) for business unit NORTH, as of 2026-03-12, to maximize net benefit in USD cents. The model must account for per-unit benefits, item and category activation fees, resource and category quantity limits, option incompatibilities and dependencies, bundle bonuses, and unit conversions for resource usage and capacity. Only the latest non-future, non-DELETE revision for each (tenant, table, record_id) is used.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation) costs, resource and category constraints, and logical (incompatibility/dependency) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items/options (`i`): All item_ref values for authorized options in business unit NORTH, after applying the selection/filtering rules.
    - Categories (`g`): All categories associated with items in NORTH.
    - Resources (`r`): All resources (e.g., labor, space, power) used by items in NORTH.
    - Bundles (`b`): All bundle bonus pairs (item_a, item_b) in NORTH.
    - Incompatibility pairs (`(i,j)`): All incompatible item pairs in NORTH.
    - Requires pairs (`(i,k)`): All requires/prerequisite pairs in NORTH.
4.  **Define Decision Variables:**
    -   `x[i]` = Integer quantity ordered of item/option `i`. Type: GRB.INTEGER. Must be 0 if unauthorized.
    -   `y[i]` = Binary variable: 1 if item/option `i` is ordered (i.e., x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `z[g]` = Binary variable: 1 if any item in category `g` is ordered, 0 otherwise. Type: GRB.BINARY.
    -   `w[b]` = Binary variable: 1 if both items in bundle pair `b` are ordered (i.e., x[item_a]>0 and x[item_b]>0), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Per-unit benefit for each item: sum of 'amount_cents' from all 'benefit' rows for item_ref `i` (from benefit tables), after filtering for latest valid revision.
    -   Item activation fee: 'activation_fee_cents' from 'item_fee' table for item_ref `i`.
    -   Category activation fee: 'activation_fee_cents' from 'category' table for category `g`.
    -   Bundle bonus: 'bonus_cents' from 'bundle' table for each bundle pair (item_a, item_b).
    -   Resource usage per unit: 'amount' and 'unit' from 'usage' table for each item/resource, with unit conversion as needed (e.g., 1000 ml = 1 liter, 60 min = 1 hour, 1000 wh = 1 kwh).
    -   Resource capacity: sum of 'amount' (with sign) from 'capacity_ledger' table for each resource, after unit conversion and filtering for latest valid revision.
    -   Category quantity limits: 'minimum_quantity' and 'maximum_quantity' from 'category' table for each category.
    -   Item quantity bounds: 'minimum_lot' and 'maximum_order' from 'item' table for each item.
    -   Authorization: 'authorized' field from 'item' table for each item (must be >0 to allow ordering).
    -   Incompatibility pairs: from 'incompatible' table (item_a, item_b).
    -   Requires pairs: from 'requires' table (item_ref, prerequisite_ref).
6.  **Formulate Objective:** Maximize total net benefit in USD cents, defined as:
    -   Sum over items: (per-unit benefit * x[i])
    -   Minus: sum of item activation fees for each item with x[i]>0 (i.e., y[i]=1)
    -   Minus: sum of category activation fees for each category with any item ordered (i.e., z[g]=1)
    -   Plus: sum of bundle bonuses for each bundle where both items are ordered (i.e., w[b]=1)
    -   (All fixed fees and bonuses are in cents; no sign reversal or shifting.)
7.  **Formulate Constraints:**
    -   **Data Selection/Filtering:** For each table, only include rows for tenant=NORTH, effective_date ≤ 2026-03-12, and for each (table, tenant, record_id), keep only the row with the highest integer revision (discard if action=DELETE). Identical retransmissions count once.
    -   **Authorization:** For each item, if 'authorized'==0, enforce x[i]=0.
    -   **Item Quantity Bounds:** For each item, enforce minimum_lot[i] * y[i] ≤ x[i] ≤ maximum_order[i] * y[i]; x[i] ≥ 0 and integer.
    -   **Category Quantity Limits:** For each category, sum of x[i] over all items in category g must satisfy minimum_quantity[g] * z[g] ≤ sum_i_in_g x[i] ≤ maximum_quantity[g] * z[g].
    -   **Category Activation:** z[g] = 1 if any x[i]>0 for i in category g; z[g]=0 otherwise.
    -   **Resource Capacity:** For each resource, sum over items of (resource usage per unit * x[i]) ≤ total available capacity for that resource (after unit conversion).
    -   **Incompatibility:** For each incompatible pair (i,j), enforce y[i] + y[j] ≤ 1 (i.e., cannot order both).
    -   **Requires/Dependencies:** For each requires pair (i,k), enforce y[i] ≤ y[k] and x[k] ≥ 1 if x[i] ≥ 1 (i.e., cannot order i unless k is also ordered in positive quantity).
    -   **Bundle Bonuses:** For each bundle (item_a, item_b), w[b] ≤ y[item_a], w[b] ≤ y[item_b], w[b] ≥ y[item_a] + y[item_b] - 1 (i.e., w[b]=1 iff both items are ordered).
    -   **Item Activation Fee:** For each item, charge activation_fee_cents only if x[i]>0 (i.e., y[i]=1).
    -   **Category Activation Fee:** For each category, charge activation_fee_cents only if z[g]=1.
    -   **Bundle Bonus:** For each bundle, award bonus_cents only if w[b]=1; zero otherwise (including if either item is unauthorized).
    -   **Variable Types:** x[i] integer ≥ 0; y[i], z[g], w[b] binary.
    -   **Unit Conversions:** When summing or comparing resource usage and capacity, convert all units to a common base (e.g., ml to liter, minute to hour, wh to kwh) as specified.
[Abstract Model Plan END]