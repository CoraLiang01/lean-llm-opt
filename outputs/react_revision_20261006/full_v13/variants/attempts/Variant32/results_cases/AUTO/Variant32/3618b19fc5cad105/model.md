## Symbolic Mathematical Model

### Sets
- $I$: set of items/options (indexed by $i$), from valid OSLO_NEW_CARS item records as of 2026-05-07.
- $C$: set of categories (indexed by $c$), from valid OSLO_NEW_CARS category records.
- $R$: set of resources (indexed by $r$), from valid OSLO_NEW_CARS resource records.
- $B$: set of valid bundle pairs $(i,j)$, from valid OSLO_NEW_CARS bundle records.
- $P$: set of incompatible pairs $(i,j)$, from valid OSLO_NEW_CARS incompatible records.
- $Q$: set of requires pairs $(i,j)$, where $i$ requires $j$.
- $S_c$: set of items in category $c$.

### Parameters
- $a_i$: 1 if item $i$ is authorized, 0 otherwise (from item.authorized).
- $l_i$: minimum lot size for item $i$ (from item.minimum_lot).
- $u_i$: maximum order for item $i$ (from item.maximum_order).
- $cat_i$: category of item $i$ (from item.category).
- $L_c$: minimum quantity for category $c$ (from category.minimum_quantity).
- $U_c$: maximum quantity for category $c$ (from category.maximum_quantity).
- $F_c$: activation fee for category $c$ (from category.activation_fee_cents).
- $f_i$: item fee for item $i$ (from item_fee.activation_fee_cents).
- $b_{ij}$: bundle bonus for $(i,j)$ (from bundle.bonus_cents).
- $r_{ir}$: resource usage per unit of item $i$ for resource $r$ (from usage.amount, converted to base units).
- $K_r$: total available resource $r$ (from capacity_ledger, sum of signed valid entries, converted to base units).
- $v_{ik}$: benefit component $k$ for item $i$ (from benefit, after currency conversion).
- $fx_{curr}$: currency conversion rate to USD cents (from fx).
- $q_i$: integer variable, quantity ordered of item $i$.

### Decision Variables
- $q_i \in \mathbb{Z}_+$: quantity of item $i$ ordered (integer, $0$ if unauthorized).
- $z_c \in \{0,1\}$: 1 if any item in category $c$ is ordered, 0 otherwise.
- $y_i \in \{0,1\}$: 1 if $q_i > 0$, 0 otherwise.
- $w_{ij} \in \{0,1\}$: 1 if both $q_i > 0$ and $q_j > 0$, 0 otherwise (for bundle bonuses).

### Objective
Maximize net benefit in USD cents:
\[
\max \left\{
\sum_{i \in I} \left( \sum_{k} v_{ik} \right) q_i
- \sum_{i \in I} f_i y_i
- \sum_{c \in C} F_c z_c
+ \sum_{(i,j) \in B} b_{ij} w_{ij}
\right\}
\]

### Constraints

#### 1. Authorization and Lot Size
\[
q_i = 0 \quad \text{if } a_i = 0
\]
\[
a_i l_i y_i \leq q_i \leq a_i u_i y_i \quad \forall i \in I
\]
\[
y_i = 1 \iff q_i > 0; \quad y_i \in \{0,1\}
\]

#### 2. Category Quantity Limits
\[
L_c z_c \leq \sum_{i \in S_c} q_i \leq U_c z_c \quad \forall c \in C
\]
\[
z_c = 1 \iff \sum_{i \in S_c} q_i > 0; \quad z_c \in \{0,1\}
\]

#### 3. Resource Capacity
\[
\sum_{i \in I} r_{ir} q_i \leq K_r \quad \forall r \in R
\]
(Resource usage and capacity must be in the same base units: liters $\to$ ml, hours $\to$ minutes, kwh $\to$ wh.)

#### 4. Incompatibility
\[
q_i + q_j \leq 1 \quad \forall (i,j) \in P
\]
(At most one of each incompatible pair can be ordered.)

#### 5. Requires Dependencies
\[
q_i \leq u_i \cdot y_j \quad \forall (i,j) \in Q
\]
(Item $i$ can be ordered only if $q_j > 0$.)

#### 6. Bundle Bonuses
\[
w_{ij} \leq y_i, \quad w_{ij} \leq y_j, \quad w_{ij} \geq y_i + y_j - 1 \quad \forall (i,j) \in B
\]
(Bundle bonus awarded once if both $i$ and $j$ are ordered.)

#### 7. Integer and Binary Domains
\[
q_i \in \mathbb{Z}_+, \quad y_i \in \{0,1\}, \quad z_c \in \{0,1\}, \quad w_{ij} \in \{0,1\}
\]

### Data Mapping

- All sets, parameters, and variables are mapped to the latest valid records as of 2026-05-07, per the selection rule:
    - For each (dealership_id, table, record_id), keep the highest integer revision not in the future, discard if DELETE.
    - Use only these records for all joins and parameter values.
- Currency conversion for each benefit component: $v_{ik} = \text{amount} \times \text{usd\_cents\_numerator} / \text{denominator}$ using the latest valid fx for the currency.
- Resource usage and capacity are converted to base units: $1$ liter $= 1000$ ml, $1$ hour $= 60$ minutes, $1$ kwh $= 1000$ wh.
- Item and category activation fees, bundle bonuses, and all fixed fees are in USD cents.
- All constraints and summations are over the valid, selected entities as described.

### Summary

This model selects integer order quantities for authorized options, subject to lot, category, resource, incompatibility, and dependency constraints, maximizing total net benefit in USD cents, including all fixed and bundle bonuses, with all data mapped as above.