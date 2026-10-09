## Symbolic Mathematical Model

### Sets
- $I$: set of item offers (from file_5_view_0, column item_ref)
- $C$: set of categories (from file_2_view_0, column category)
- $R$: set of resources (from file_1_view_0, column resource)
- $B$: set of bundle bonus pairs (from file_0_view_0, columns item_a, item_b)
- $P$: set of incompatible item pairs (from file_4_view_0, columns item_a, item_b)
- $Q$: set of requires dependencies (from file_7_view_0, columns item_ref, prerequisite_ref)

### Parameters
- $u_i$: unit benefit in cents for item $i$ (file_5_view_0, unit_benefit_cents)
- $f_i$: fixed item fee in cents for item $i$ (file_5_view_0, item_fee_cents)
- $a_i$: 1 if item $i$ is authorized, 0 otherwise (file_5_view_0, authorized)
- $l_i$: minimum lot for item $i$ (file_5_view_0, minimum_lot)
- $m_i$: maximum order for item $i$ (file_5_view_0, maximum_order)
- $cat_i$: category of item $i$ (file_5_view_0, category)
- $catmin_c$: minimum quantity for category $c$ (file_2_view_0, minimum_quantity)
- $catmax_c$: maximum quantity for category $c$ (file_2_view_0, maximum_quantity)
- $catfee_c$: activation fee for category $c$ (file_2_view_0, activation_fee_cents)
- $b_{ij}$: bundle bonus in cents for pair $(i,j)\in B$ (file_0_view_0, bonus_cents)
- $res_{ir}$: usage of resource $r$ per unit of item $i$ (file_8_view_0, amount; missing pairs are zero)
- $cap_r$: total available for resource $r$ (sum of file_1_view_0, entry=opening, amount for $r$ plus all reservation entries for $r$)
- $inc_{ij}$: 1 if $(i,j)\in P$ or $(j,i)\in P$, 0 otherwise (from file_4_view_0)
- $req_{ij}$: 1 if $(i,j)\in Q$, 0 otherwise (from file_7_view_0)

### Decision Variables
- $x_i \in \mathbb{Z}_{\geq 0}$: quantity of item $i$ to order
- $z_i \in \{0,1\}$: 1 if $x_i > 0$, 0 otherwise (item activation)
- $y_c \in \{0,1\}$: 1 if any item in category $c$ is selected, 0 otherwise (category activation)
- $w_{ij} \in \{0,1\}$: 1 if both $x_i > 0$ and $x_j > 0$ for bundle $(i,j)\in B$, 0 otherwise

### Objective
Maximize net benefit in USD cents:
\[
\max \left\{
\sum_{i\in I} u_i x_i
- \sum_{i\in I} f_i z_i
- \sum_{c\in C} catfee_c\, y_c
+ \sum_{(i,j)\in B} b_{ij} w_{ij}
\right\}
\]

### Constraints

#### 1. Authorization and Lot/Order Bounds
\[
\forall i\in I:
\begin{cases}
x_i = 0 & \text{if } a_i = 0 \\
l_i z_i \leq x_i \leq m_i z_i & \text{if } a_i = 1 \\
z_i \in \{0,1\} \\
x_i \in \mathbb{Z}_{\geq 0}
\end{cases}
\]

#### 2. Category Quantity Limits (Unconditional)
\[
\forall c\in C:
\quad
catmin_c \leq \sum_{i\in I: cat_i = c} x_i \leq catmax_c
\]

#### 3. Category Activation
\[
\forall c\in C:
\quad
y_c \geq z_i \quad \forall i\in I: cat_i = c
\]
\[
y_c \in \{0,1\}
\]

#### 4. Resource Capacity
\[
\forall r\in R:
\quad
\sum_{i\in I} res_{ir} x_i \leq cap_r
\]

#### 5. Incompatible Pairs
\[
\forall (i,j)\in P:
\quad
x_i = 0 \quad \text{or} \quad x_j = 0
\]
(equivalently: $z_i + z_j \leq 1$)

#### 6. Requires Dependencies
\[
\forall (i,j)\in Q:
\quad
x_i \leq m_i z_j
\]
($x_i > 0$ only if $x_j > 0$)

#### 7. Bundle Bonuses
\[
\forall (i,j)\in B:
\quad
w_{ij} \leq z_i
\]
\[
w_{ij} \leq z_j
\]
\[
w_{ij} \geq z_i + z_j - 1
\]
\[
w_{ij} \in \{0,1\}
\]

### Data Mapping

- file_0_view_0: bundle bonuses, $(i,j)$ and $b_{ij}$
- file_1_view_0: resource capacity ledger, $cap_r$ (sum opening and reservation for each $r$)
- file_2_view_0: category limits and activation fees, $catmin_c$, $catmax_c$, $catfee_c$
- file_3_view_0: item identity, for mapping display names if needed
- file_4_view_0: incompatible item pairs, $P$
- file_5_view_0: item offers and constraints, $I$, $u_i$, $f_i$, $a_i$, $l_i$, $m_i$, $cat_i$
- file_7_view_0: requires dependencies, $Q$
- file_8_view_0: resource usage per item, $res_{ir}$

All indices, parameters, and constraints are defined directly from the supplied tables. All units are in cents and base units as provided. The objective is the maximum net benefit in USD cents.