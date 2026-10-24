[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the integer replenishment quantities for each authorized option (item) in business unit NORTH, as of 2026-03-12, that maximize net benefit (in USD cents). The model must account for per-unit benefit, item and category activation fees, bundle bonuses, resource and category quantity limits, minimum/maximum order sizes, authorization, incompatibilities, and requires dependencies. All data tables must be filtered to the latest revision on or before the as-of date, per (tenant, table, record_id), discarding DELETEs and ignoring future events and retransmissions.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation fee) and logical (dependency/incompatibility) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items (options) `i` (from item tables, after as-of/revision filtering)
    - Categories `g` (from category tables, after filtering)
    - Resources `r` (from usage/capacity tables)
    - Bundles `b` (from bundle tables)
    - Incompatible pairs `(i,j)` (from incompatible tables)
    - Requires pairs `(i,k)` (from requires tables)
4.  **Define Decision Variables:**
    - `q[i]` = integer quantity ordered of item `i`. Type: GRB.INTEGER.
    - `y[i]` = 1 if item `i` is selected (i.e., `q[i] > 0`), 0 otherwise. Type: GRB.BINARY.
    - `z[g]` = 1 if any item in category `g` is selected, 0 otherwise. Type: GRB.BINARY.
    - `w[b]` = 1 if both items in bundle `b` are selected, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    - Per-unit benefit: sum of `amount_cents` from all benefit table rows for item `i` (after as-of/revision filtering).
    - Item activation fee: `activation_fee_cents` from item_fee table for item `i` (after filtering).
    - Category activation fee: `activation_fee_cents` from category table for category `g` (after filtering).
    - Bundle bonus: `bonus_cents` from bundle table for bundle `b` (after filtering).
    - Minimum/maximum order: `minimum_lot`, `maximum_order` from item table for item `i`.
    - Authorization: `authorized` from item table for item `i` (must be positive for eligibility).
    - Category quantity limits: `minimum_quantity`, `maximum_quantity` from category table for category `g`.
    - Resource usage per unit: `amount` and `unit` from usage table for item `i` and resource `r`.
    - Resource capacity: sum of `amount` (converted to common units) from capacity_ledger table for resource `r` (after filtering).
    - Incompatible pairs: item pairs from incompatible table (after filtering).
    - Requires dependencies: pairs from requires table (after filtering).
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    - Sum over items: (per-unit benefit) × `q[i]`
    - Minus: sum of item activation fees for each item with `y[i]=1`
    - Minus: sum of category activation fees for each category with `z[g]=1`
    - Plus: sum of bundle bonuses for each bundle with `w[b]=1`
7.  **Formulate Constraints:**
    - **Item eligibility and order bounds:** For each item `i`:
        - If `authorized[i] > 0`: `minimum_lot[i] * y[i] ≤ q[i] ≤ maximum_order[i] * y[i]`, and `q[i] ≥ 0`, integer.
        - If `authorized[i] ≤ 0`: `q[i] = 0`, `y[i] = 0`.
        - `y[i] = 1` iff `q[i] > 0`; `y[i] = 0` iff `q[i] = 0`.
    - **Category activation and quantity limits:** For each category `g`:
        - `z[g] = 1` iff any `y[i]=1` for item `i` in category `g`.
        - `z[g] ≤ sum_{i in g} y[i]`; for all `i` in `g`, `y[i] ≤ z[g]`.
        - Category quantity limits: `minimum_quantity[g] ≤ sum_{i in g} q[i] ≤ maximum_quantity[g]`.
    - **Resource capacity constraints:** For each resource `r`:
        - Convert all item usage and capacity to a common unit (e.g., ml for space, wh for power, minutes for labor).
        - `sum_{i} (usage[i,r] per unit × q[i]) ≤ total capacity[r]` (capacity is sum of all relevant capacity_ledger entries, after filtering).
    - **Incompatibility constraints:** For each incompatible pair `(i,j)`:
        - `y[i] + y[j] ≤ 1` (cannot select both).
    - **Requires dependencies:** For each requires pair `(i,k)`:
        - `q[i] > 0` ⇒ `q[k] > 0` (i.e., `y[i] ≤ y[k]`).
    - **Bundle bonuses:** For each bundle `b` with items `(i,j)`:
        - `w[b] = 1` iff `y[i]=1` and `y[j]=1`; `w[b] ≤ y[i]`, `w[b] ≤ y[j]`, `w[b] ≥ y[i] + y[j] - 1`.
        - Award bonus only if both items are selected and authorized.
    - **Integrality:** All `q[i]` are integer, all flags are binary.
[Abstract Model Plan END]