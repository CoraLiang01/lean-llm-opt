## Symbolic Mathematical Model

### Sets
- $I$: set of item options (item_ref) (from file_7_view_0)
- $R$: set of resources (platforms: PC, CONSOLE, MOBILE) (from file_2_view_0)
- $G$: set of categories (from file_3_view_0)
- $B$: set of bundle pairs (from file_1_view_0)
- $C$: set of currencies (from file_4_view_0)
- $E$: set of incompatible pairs (from file_6_view_0)
- $Q$: set of requires pairs (from file_10_view_0)

### Parameters

#### Item Option Parameters (from file_7_view_0)
- $a_i$: 1 if item $i$ is authorized, 0 otherwise
- $g_i$: category of item $i$
- $l_i$: minimum_lot for item $i$
- $u_i$: maximum_order for item $i$
- $s_i$: location_id (platform) for item $i$

#### Benefit Calculation (from file_0_view_0, file_4_view_0)
- $K_i$: set of benefit components for item $i$
- For each component $k \in K_i$:
    - $A_{ik}$: amount
    - $C_{ik}$: currency
    - $N_{C_{ik}}$: usd_cents_numerator for currency $C_{ik}$
    - $D_{C_{ik}}$: denominator for currency $C_{ik}$
    - $F_{ik}$: conversion factor for $C_{ik}$, $F_{ik} = N_{C_{ik}} / D_{C_{ik}}$
- $b_i = \sum_{k \in K_i} A_{ik} \cdot F_{ik}$: total per-unit benefit in USD cents for item $i$

#### Item Fees (from file_8_view_0)
- $f_i$: item_fee (activation_fee_cents) for item $i$

#### Resource Usage (from file_11_view_0)
- $u_{ir}$: per-unit usage of resource $r$ by item $i$ (in MB; convert GB to MB by $u_{ir} = \text{amount} \times 1000$ if unit is GB, else use as is; 0 if not listed)

#### Resource Capacity (from file_2_view_0)
- $C_r$: total available capacity for resource $r$ (sum of all ledger entries for $r$)

#### Category Limits (from file_3_view_0)
- $L_g$: minimum_quantity for category $g$
- $U_g$: maximum_quantity for category $g$
- $F_g$: activation_fee_cents for category $g$

#### Bundle Bonuses (from file_1_view_0)
- For each bundle $(i,j) \in B$, $q_{ij}$: bonus_cents

#### Incompatibility (from file_6_view_0)
- $E$: set of unordered pairs $(i,j)$, $i < j$, that are incompatible

#### Requires (from file_10_view_0)
- $Q$: set of ordered pairs $(i,pr)$, $i$ requires $pr$

### Decision Variables
- $x_i \in \mathbb{Z}_+$: number of units of item $i$ to allocate
- $y_i \in \{0,1\}$: 1 if $x_i > 0$, 0 otherwise (item activation)
- $z_g \in \{0,1\}$: 1 if any $x_i > 0$ for $i$ in category $g$ (category activation)
- $w_{ij} \in \{0,1\}$: 1 if both $x_i > 0$ and $x_j > 0$ (bundle activation)

### Objective
Maximize net benefit in USD cents:
\[
\max \Bigg\{
\sum_{i \in I} \left[ b_i x_i - f_i y_i \right]
+ \sum_{(i,j) \in B} q_{ij} w_{ij}
- \sum_{g \in G} F_g z_g
\Bigg\}
\]

### Constraints

#### Authorization and Integer Bounds
\[
x_i = 0 \quad \forall i \in I \text{ with } a_i = 0
\]
\[
x_i \in \{0\} \cup [l_i, u_i] \cap \mathbb{Z} \quad \forall i \in I \text{ with } a_i = 1
\]

#### Item Activation
\[
y_i \geq \frac{x_i}{u_i} \quad \forall i \in I
\]
\[
y_i \leq \frac{x_i}{l_i} \quad \forall i \in I \text{ with } a_i = 1
\]
\[
y_i \in \{0,1\} \quad \forall i \in I
\]

#### Resource Capacity (per platform)
\[
\sum_{i \in I} u_{ir} x_i \leq C_r \quad \forall r \in R
\]

#### Category Quantity Limits and Activation
\[
L_g \leq \sum_{i \in I: g_i = g} x_i \leq U_g \quad \forall g \in G
\]
\[
z_g \geq y_i \quad \forall i \in I: g_i = g
\]
\[
z_g \in \{0,1\} \quad \forall g \in G
\]

#### Incompatibility
\[
y_i + y_j \leq 1 \quad \forall (i,j) \in E
\]

#### Requires
\[
y_i \leq y_{pr} \quad \forall (i,pr) \in Q
\]

#### Bundle Bonuses
\[
w_{ij} \leq y_i, \quad w_{ij} \leq y_j, \quad w_{ij} \geq y_i + y_j - 1 \quad \forall (i,j) \in B
\]
\[
w_{ij} \in \{0,1\} \quad \forall (i,j) \in B
\]

#### Zero for Unauthorized
\[
x_i = 0, \quad y_i = 0 \quad \forall i \in I \text{ with } a_i = 0
\]

### Data Mapping

- file_0_view_0: benefit components per item_ref, with amount and currency
- file_4_view_0: currency conversion (usd_cents_numerator, denominator) for each currency
- file_8_view_0: item_ref to item_fee (activation_fee_cents)
- file_11_view_0: item_ref, resource, amount, unit (GB or MB) for per-unit usage
- file_2_view_0: resource, amount, unit (MB), sum for each resource gives $C_r$
- file_7_view_0: item_ref, authorized, category, minimum_lot, maximum_order, location_id
- file_3_view_0: category, minimum_quantity, maximum_quantity, activation_fee_cents
- file_1_view_0: item_a, item_b, bonus_cents (bundle bonuses)
- file_6_view_0: item_a, item_b (incompatible pairs)
- file_10_view_0: item_ref, prerequisite_ref (requires pairs)

### Notes

- All sums, indices, and constraints are over the current rows/entities in the respective tables.
- All units are in USD cents, MB, and integer units as per the data.
- Only authorized items may be selected; unauthorized items must have zero allocation.
- Each item option is for a specific platform; memory capacity is enforced per platform.
- Each category's activation fee is charged once if any item in that category is selected.
- Each item's activation fee is charged once if any quantity is selected.
- Each bundle bonus is counted once if both items are selected.
- Incompatible pairs cannot both be selected.
- Requires pairs enforce that if an item is selected, its prerequisite must also be selected (but not necessarily in any proportion).
- All variable domains, bounds, and constraints are enforced exactly as described.

---

This model, with the above Data Mapping, fully encodes the allocation and licensing optimization as described in the current data and user query.