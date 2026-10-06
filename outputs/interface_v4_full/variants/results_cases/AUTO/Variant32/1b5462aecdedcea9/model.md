[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine, for the Oslo dealership as of 2026-05-07, the vehicle order (i.e., a selection of options and their integer quantities) that yields the largest net benefit in USD cents. This must respect all business rules: only valid, non-deleted, latest-revision records as of the date; currency conversion; item and category limits; resource and compatibility constraints; option authorization; bundle bonuses; item and category activation fees; and requires/incompatible dependencies. The model must select integer lot sizes for authorized options (within min/max), zero for unauthorized, and apply all fixed and per-unit costs/bonuses as described.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed charges, resource constraints, and logical (compatibility/dependency) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Options/Items (`i`): All valid OSLO_NEW_CARS options (from item tables after filtering).
    - Categories (`g`): All valid OSLO_NEW_CARS categories.
    - Resources (`r`): All valid OSLO_NEW_CARS resources (e.g., space, power, labor).
    - Bundles (`b`): All valid OSLO_NEW_CARS bundle bonus pairs.
    - Incompatible pairs (`(i,j)`): All valid OSLO_NEW_CARS incompatible option pairs.
    - Requires pairs (`(i,k)`): All valid OSLO_NEW_CARS requires dependencies.
4.  **Define Decision Variables:**
    -   `q[i]` = Integer quantity ordered of option/item `i`. Type: GRB.INTEGER. Must be 0 if unauthorized, else between minimum_lot and maximum_order.
    -   `z[i]` = 1 if option/item `i` is selected (i.e., q[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `y[g]` = 1 if any option in category `g` is selected (i.e., category is used), 0 otherwise. Type: GRB.BINARY.
    -   `w[b]` = 1 if both options in bundle `b` are selected (i.e., both q[i_a]>0 and q[i_b]>0), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   **Benefit per unit (USD cents):** For each option, sum all valid benefit components (from benefit tables), converting each amount to USD cents using the latest valid fx rate as of 2026-05-07 (amount * usd_cents_numerator / denominator).
    -   **Item activation fee (USD cents):** For each option, from item_fee tables (activation_fee_cents).
    -   **Category activation fee (USD cents):** For each category, from category tables (activation_fee_cents).
    -   **Bundle bonus (USD cents):** For each bundle, from bundle tables (bonus_cents).
    -   **Resource usage per unit:** For each option and resource, from usage tables (amount, unit), converted to base units (ml, minute, wh).
    -   **Resource capacity:** For each resource, sum all valid capacity_ledger entries (amount, unit), converted to base units.
    -   **Option authorization, min/max lot:** From item tables (authorized, minimum_lot, maximum_order).
    -   **Category min/max quantity:** From category tables (minimum_quantity, maximum_quantity).
    -   **Incompatible pairs:** From incompatible tables (item_a, item_b).
    -   **Requires dependencies:** From requires tables (item_ref, prerequisite_ref).
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    -   Sum over all options: (benefit per unit in USD cents) * q[i]
    -   Minus: sum of item activation fees for each option with q[i]>0 (charge once per option)
    -   Minus: sum of category activation fees for each category used (charge once per category)
    -   Plus: sum of bundle bonuses for each bundle where both options are selected (award once per bundle)
7.  **Formulate Constraints:**
    -   **Option authorization and lot size:** For each option `i`, if authorized=0 then q[i]=0; if authorized>0 then minimum_lot ≤ q[i] ≤ maximum_order or q[i]=0.
    -   **Linking q[i] and z[i]:** For each option, q[i] ≥ minimum_lot * z[i], q[i] ≤ maximum_order * z[i], z[i] ∈ {0,1}.
    -   **Category quantity limits:** For each category `g`, sum of q[i] over all options in `g` must be between minimum_quantity and maximum_quantity (if any option in `g` is selected).
    -   **Category activation:** For each category, y[g] = 1 if any q[i]>0 for i in g; y[g] ∈ {0,1}.
    -   **Resource capacity:** For each resource `r`, sum over all options of (resource usage per unit for r) * q[i] ≤ total available capacity for r (all in base units: 1000 ml/liter, 60 min/hour, 1000 wh/kwh).
    -   **Incompatible pairs:** For each incompatible pair (i,j), z[i] + z[j] ≤ 1 (cannot select both).
    -   **Requires dependencies:** For each requires pair (i,k), q[i] > 0 ⇒ q[k] > 0 (can be modeled as q[i] ≤ M * z[k], with M large enough).
    -   **Bundle bonuses:** For each bundle (i_a, i_b), w[b] = 1 if z[i_a]=1 and z[i_b]=1; w[b] ∈ {0,1}.
    -   **Category activation fee:** For each category, charge activation fee only if y[g]=1.
    -   **Item activation fee:** For each option, charge activation fee only if z[i]=1.
    -   **Bundle bonus:** For each bundle, award bonus only if w[b]=1.
    -   **All variables:** q[i] ∈ {0} ∪ [minimum_lot, maximum_order] (integers), z[i], y[g], w[b] ∈ {0,1}.
[Abstract Model Plan END]