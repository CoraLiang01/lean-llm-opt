[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the integer replenishment quantities for each authorized option (item) for business unit NORTH, using only records effective on or before 2026-03-12, so as to maximize net benefit (in USD cents). The model must account for per-unit benefit, item and category activation fees, bundle bonuses, resource and category quantity limits, option incompatibilities and dependencies, and unit conversions for resource usage and capacity.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation fee) and logical (dependency/incompatibility) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items (options) \( i \) available for replenishment.
    - Categories \( g \) to which items belong.
    - Resources \( r \) (e.g., space, labor, power).
    - Bundles \( b \) (pairs of items with a bonus).
    - Incompatible pairs \( (i, j) \).
    - Requires pairs \( (i, k) \) (item \( i \) requires \( k \)).
4.  **Define Decision Variables:**
    -   \( x[i] \) = Integer quantity ordered of item \( i \). Type: GRB.INTEGER.
    -   \( y[i] \) = 1 if item \( i \) is ordered (i.e., \( x[i] > 0 \)), 0 otherwise. Type: GRB.BINARY.
    -   \( z[g] \) = 1 if any item in category \( g \) is ordered, 0 otherwise. Type: GRB.BINARY.
    -   \( w[b] \) = 1 if both items in bundle \( b \) are ordered, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Per-unit benefit for each item \( i \): sum of 'amount_cents' from 'benefit' table, grouped by 'item_ref'.
    -   Item activation fee: 'activation_fee_cents' from 'item_fee' table, mapped by 'item_ref'.
    -   Category activation fee: 'activation_fee_cents' from 'category' table, mapped by 'category'.
    -   Bundle bonus: 'bonus_cents' from 'bundle' table, mapped by item pairs.
    -   Resource usage per unit: 'amount' and 'unit' from 'usage' table, mapped by 'item_ref' and 'resource', with unit conversion (e.g., liter to ml, hour to minute, kwh to wh).
    -   Resource capacity: sum of 'amount' from 'capacity_ledger' table (all entries up to cutoff date), per resource, with unit conversion.
    -   Item bounds: 'minimum_lot', 'maximum_order', and 'authorized' from 'item' table, mapped by 'item_ref'.
    -   Category bounds: 'minimum_quantity', 'maximum_quantity' from 'category' table, mapped by 'category'.
    -   Incompatibilities: pairs from 'incompatible' table.
    -   Requires dependencies: pairs from 'requires' table.
6.  **Formulate Objective:** Maximize total net benefit, defined as:
    -   Sum over items: (per-unit benefit) × (quantity ordered)
    -   Minus: sum of item activation fees for each item with positive quantity
    -   Minus: sum of category activation fees for each category with any item ordered
    -   Plus: sum of bundle bonuses for each bundle where both items are ordered
    All terms are in USD cents.
7.  **Formulate Constraints:**
    -   **Item Authorization and Bounds:** For each item \( i \), if 'authorized' > 0, enforce \( x[i] = 0 \) or \( minimum\_lot[i] \leq x[i] \leq maximum\_order[i] \); if 'authorized' = 0, enforce \( x[i] = 0 \).
    -   **Item Activation Linking:** For each item \( i \), \( y[i] = 1 \) if \( x[i] > 0 \), else 0; enforce \( x[i] \leq maximum\_order[i] \cdot y[i] \) and \( x[i] \geq minimum\_lot[i] \cdot y[i] \) (if authorized).
    -   **Category Activation Linking:** For each category \( g \), \( z[g] = 1 \) if any \( x[i] > 0 \) for \( i \) in \( g \), else 0.
    -   **Category Quantity Limits:** For each category \( g \), sum of \( x[i] \) over items in \( g \) must satisfy \( minimum\_quantity[g] \leq \sum_{i \in g} x[i] \leq maximum\_quantity[g] \).
    -   **Resource Capacity:** For each resource \( r \), sum over items of (resource usage per unit, converted to ledger units) × \( x[i] \) ≤ total available capacity for \( r \) (converted to same units).
    -   **Incompatibility:** For each incompatible pair \( (i, j) \), at most one of \( x[i], x[j] \) is positive: \( y[i] + y[j] \leq 1 \).
    -   **Requires Dependency:** For each requires pair \( (i, k) \), \( x[i] > 0 \) implies \( x[k] > 0 \); enforce \( y[i] \leq y[k] \).
    -   **Bundle Bonus Linking:** For each bundle \( b \) with items \( (i, j) \), \( w[b] = 1 \) if both \( y[i] = 1 \) and \( y[j] = 1 \), else 0; enforce \( w[b] \leq y[i] \), \( w[b] \leq y[j] \), \( w[b] \geq y[i] + y[j] - 1 \).
    -   **Integrality:** All \( x[i] \) are integer, all \( y[i], z[g], w[b] \) are binary.
[Abstract Model Plan END]