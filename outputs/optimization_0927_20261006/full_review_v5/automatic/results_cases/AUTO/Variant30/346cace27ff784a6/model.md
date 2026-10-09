[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine, for business unit NORTH as of 2026-03-12, the integer replenishment quantities for each authorized option (item) that maximize net benefit in USD cents. The plan must account for per-unit benefits, item and category activation fees, resource and category quantity limits, option incompatibilities and dependencies, bundle bonuses, and unit conversions. All data must be filtered by the latest revision on or before the as-of date, excluding DELETEs, and identical retransmissions count once. This selection rule applies to every table before any join or aggregation.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation fee) and logical (compatibility/dependency) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items (options) `i` (from item tables, after filtering)
    - Categories `g` (from category tables, after filtering)
    - Resources `r` (from resource usage/capacity tables)
    - Bundles `b` (from bundle tables)
    - Incompatible pairs `(i,j)` and requires pairs `(i,k)` (from respective tables)
4.  **Define Decision Variables:**
    - `q[i]` = integer quantity ordered of item `i`. Type: GRB.INTEGER, with bounds as below.
    - `z[i]` = binary, 1 if item `i` is selected (i.e., `q[i] > 0`), 0 otherwise. Type: GRB.BINARY.
    - `y[g]` = binary, 1 if any item in category `g` is selected, 0 otherwise. Type: GRB.BINARY.
    - `w[b]` = binary, 1 if both items in bundle `b` are selected, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    - Per-unit benefit: sum of all `amount_cents` for each item from all benefit tables (after filtering and summing by item).
    - Item activation fee: `activation_fee_cents` from item_fee tables (after filtering, mapped by item).
    - Category activation fee: `activation_fee_cents` from category tables (after filtering, mapped by category).
    - Bundle bonus: `bonus_cents` from bundle tables (after filtering, mapped by bundle).
    - Resource usage per unit: `amount` and `unit` from usage tables (after filtering, mapped by item and resource).
    - Resource capacity: sum of all `amount` (with sign) for each resource from capacity_ledger tables (after filtering, converted to canonical units).
    - Item bounds: `minimum_lot`, `maximum_order`, and `authorized` from item tables (after filtering, mapped by item).
    - Category quantity bounds: `minimum_quantity`, `maximum_quantity` from category tables (after filtering, mapped by category).
    - Incompatible pairs: from incompatible tables (after filtering, mapped by item pairs).
    - Requires pairs: from requires tables (after filtering, mapped by item and prerequisite).
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    - Sum over items: (per-unit benefit) × `q[i]`
    - Minus: sum over selected items: item activation fee × `z[i]` (only if `q[i]>0`)
    - Minus: sum over used categories: category activation fee × `y[g]` (only if any item in `g` is selected)
    - Plus: sum over bundles: bundle bonus × `w[b]` (only if both items in bundle are selected)
7.  **Formulate Constraints:**
    - **Data Preprocessing:** For every table, before any join or sum, filter to tenant=NORTH, effective_date ≤ 2026-03-12, keep only the highest integer revision per (tenant, table, record_id), and discard if that revision is DELETE. Identical retransmissions count once.
    - **Item Authorization and Bounds:** For each item `i`:
        - If `authorized[i] > 0`: `minimum_lot[i] * z[i] ≤ q[i] ≤ maximum_order[i] * z[i]`, `q[i]` integer ≥ 0, `z[i]` binary, and `q[i]=0` iff `z[i]=0`.
        - If `authorized[i] = 0`: `q[i] = 0`, `z[i] = 0`.
    - **Category Quantity Limits:** For each category `g`:
        - Let `S_g` be the set of items in category `g`.
        - `minimum_quantity[g] * y[g] ≤ sum_{i in S_g} q[i] ≤ maximum_quantity[g] * y[g]`
        - For all `i in S_g`: `z[i] ≤ y[g]`; for all `g`: `y[g] ≤ sum_{i in S_g} z[i]`
    - **Resource Capacity Constraints:** For each resource `r`:
        - Convert all item usage and capacity to canonical units (liter→ml, hour→minute, kwh→wh as specified).
        - `sum_{i} (resource_usage_per_unit[i,r] × q[i]) ≤ total_capacity[r]`
    - **Incompatibility Constraints:** For each incompatible pair `(i,j)`:
        - `z[i] + z[j] ≤ 1`
    - **Requires Constraints:** For each requires pair `(i,k)`:
        - `z[i] ≤ z[k]` and `q[i] > 0` ⇒ `q[k] > 0` (enforced by `z[i] ≤ z[k]`)
    - **Bundle Bonus Activation:** For each bundle `b` with items `(i,j)`:
        - `w[b] ≤ z[i]`, `w[b] ≤ z[j]`, `w[b] ≥ z[i] + z[j] - 1`
    - **Variable Domains:** All variables as defined above; all quantities integer, all flags binary.
[Abstract Model Plan END]