[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the integer replenishment quantities for each authorized option (item) for business unit NORTH, as of 2026-03-12, that maximize net benefit in USD cents. The model must account for per-unit benefit (sum of signed benefit components), item and category activation fees, bundle bonuses, resource and category quantity limits, minimum/maximum order sizes, authorization, incompatibilities, and requires dependencies. All data tables must be filtered for NORTH, as-of 2026-03-12, using the highest integer revision per (tenant, table, record_id), discarding records if that revision is DELETE, and counting identical retransmissions only once.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation fee) and logical (dependency/incompatibility) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items (options) `i` (from item tables, filtered as above)
    - Categories `g` (from category tables, filtered as above)
    - Resources `r` (from resource/usage/capacity tables, filtered as above)
    - Bundles `b` (from bundle tables, filtered as above)
    - Incompatible pairs `(i,j)` (from incompatible tables, filtered as above)
    - Requires pairs `(i,k)` (from requires tables, filtered as above)
4.  **Define Decision Variables:**
    -   `q[i]` = integer quantity ordered of item `i`. Type: GRB.INTEGER, domain: [0, maximum_order[i]].
    -   `z[i]` = binary, 1 if item `i` is selected (i.e., q[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `y[g]` = binary, 1 if any item in category `g` is selected, 0 otherwise. Type: GRB.BINARY.
    -   `w[b]` = binary, 1 if both items in bundle `b` are selected (i.e., both q[item_a[b]] > 0 and q[item_b[b]] > 0), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Per-unit benefit: sum of all `amount_cents` for each item from all benefit tables (after filtering).
    -   Item activation fee: `activation_fee_cents` from item_fee tables (after filtering), applied once per item with q[i] > 0.
    -   Category activation fee: `activation_fee_cents` from category tables (after filtering), applied once per category with any selected item.
    -   Bundle bonus: `bonus_cents` from bundle tables (after filtering), awarded once per bundle if both items are selected.
    -   Minimum/maximum order: `minimum_lot`, `maximum_order` from item tables (after filtering).
    -   Authorization: `authorized` from item tables (after filtering); only items with authorized > 0 may have q[i] > 0.
    -   Resource usage per unit: `amount` and `unit` from usage tables (after filtering), mapped to items and resources.
    -   Resource capacity: sum of signed `amount` (converted to canonical units) from capacity_ledger tables (after filtering), per resource.
    -   Category quantity limits: `minimum_quantity`, `maximum_quantity` from category tables (after filtering), per category.
    -   Incompatible pairs: from incompatible tables (after filtering), as unordered pairs of items.
    -   Requires dependencies: from requires tables (after filtering), as (item, prerequisite) pairs.
6.  **Formulate Objective:** Maximize total net benefit in USD cents, defined as:
    -   Sum over items: (per-unit benefit[i] * q[i])
    -   Minus sum over selected items: item activation_fee[i] * z[i]
    -   Minus sum over used categories: category activation_fee[g] * y[g]
    -   Plus sum over awarded bundles: bundle_bonus[b] * w[b]
7.  **Formulate Constraints:**
    -   **Item selection and order size:**
        - For each item i: q[i] = 0 if not authorized; else, minimum_lot[i] * z[i] ≤ q[i] ≤ maximum_order[i] * z[i]; z[i] ∈ {0,1}.
    -   **Category activation:**
        - For each item i in category g: z[i] ≤ y[g]; for each category g: y[g] ≤ sum_{i in g} z[i].
    -   **Category quantity limits:**
        - For each category g: minimum_quantity[g] ≤ sum_{i in g} q[i] ≤ maximum_quantity[g].
    -   **Resource capacity:**
        - For each resource r: sum_{i} (resource_usage_per_unit[i,r] * q[i]) ≤ resource_capacity[r], with all quantities converted to canonical units (e.g., ml, wh, minutes).
    -   **Incompatibility:**
        - For each incompatible pair (i,j): z[i] + z[j] ≤ 1.
    -   **Requires dependencies:**
        - For each (i,k): q[i] ≤ maximum_order[i] * z[k]; equivalently, z[i] ≤ z[k]; and q[k] ≥ 1 if q[i] ≥ 1.
    -   **Bundle bonuses:**
        - For each bundle b with items (i,j): w[b] ≤ z[i], w[b] ≤ z[j], w[b] ≥ z[i] + z[j] - 1; w[b] ∈ {0,1}.
    -   **Variable domains:**
        - All q[i] are integer, z[i], y[g], w[b] are binary.
[Abstract Model Plan END]