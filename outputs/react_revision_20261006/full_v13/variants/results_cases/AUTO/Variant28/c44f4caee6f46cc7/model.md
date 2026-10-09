Mathematical Model for CENTRAL_FRESH Produce Order

Sets:
- $I$: set of authorized items (from file_5_view_0, column item_ref)
- $C$: set of categories (from file_2_view_0, column category)
- $R$: set of resources (from file_1_view_0, column resource)
- $B$: set of bundle bonus pairs (from file_0_view_0, columns item_a, item_b)
- $P$: set of incompatible pairs (from file_4_view_0, columns item_a, item_b)
- $Q$: set of prerequisite pairs (from file_7_view_0, columns item_ref, prerequisite_ref)

Parameters:
- $u_i$: unit_benefit_cents for item $i$ (file_5_view_0)
- $f_i$: item_fee_cents for item $i$ (file_5_view_0)
- $cat(i)$: category of item $i$ (file_5_view_0)
- $min_i$, $max_i$: minimum_lot, maximum_order for item $i$ (file_5_view_0)
- $min_c$, $max_c$: minimum_quantity, maximum_quantity for category $c$ (file_2_view_0)
- $F_c$: activation_fee_cents for category $c$ (file_2_view_0)
- $a_{ir}$: per-unit usage of resource $r$ by item $i$ (file_8_view_0, amount, converted to base units)
- $cap_r$: total available capacity for resource $r$ (sum of file_1_view_0, amount, converted to base units)
- $b_{ij}$: bonus_cents for bundle $(i,j)$ (file_0_view_0)
- $inc_{ij}$: 1 if $(i,j)$ is an incompatible pair, 0 otherwise (file_4_view_0)
- $req_{ij}$: 1 if $i$ requires $j$ as a prerequisite, 0 otherwise (file_7_view_0)

Decision Variables:
- $x_i \in \mathbb{Z}_+$: number of cases of item $i$ to order ($x_i = 0$ or $min_i \leq x_i \leq max_i$)
- $y_i \in \{0,1\}$: 1 if $x_i > 0$, 0 otherwise (item activation)
- $z_c \in \{0,1\}$: 1 if any item in category $c$ is ordered, 0 otherwise (category activation)
- $w_{ij} \in \{0,1\}$: 1 if both $x_i > 0$ and $x_j > 0$ for bundle $(i,j)$, 0 otherwise (bundle activation)

Objective:
Maximize net benefit in USD cents:
\[
\max \left\{
\sum_{i \in I} u_i x_i
- \sum_{i \in I} f_i y_i
+ \sum_{(i,j) \in B} b_{ij} w_{ij}
- \sum_{c \in C} F_c z_c
\right\}
\]

Subject to:

1. Item order bounds and activation:
\[
x_i = 0 \quad \text{or} \quad min_i \leq x_i \leq max_i \qquad \forall i \in I
\]
\[
y_i = \begin{cases}
1 & \text{if } x_i > 0 \\
0 & \text{if } x_i = 0
\end{cases} \qquad \forall i \in I
\]
(Enforced via: $x_i \leq max_i y_i$, $x_i \geq min_i y_i$, $y_i \in \{0,1\}$)

2. Category quantity and activation:
\[
min_c \leq \sum_{i \in I: cat(i)=c} x_i \leq max_c \qquad \forall c \in C
\]
\[
z_c \geq y_i \qquad \forall i \in I: cat(i)=c
\]
($z_c$ can be set as $z_c = \max_{i:cat(i)=c} y_i$)

3. Resource capacity (convert all units to base: 1000 ml = 1 liter, 60 min = 1 hour, 1000 wh = 1 kwh):
\[
\sum_{i \in I} a_{ir} x_i \leq cap_r \qquad \forall r \in R
\]

4. Incompatibility:
\[
y_i + y_j \leq 1 \qquad \forall (i,j) \in P
\]

5. Prerequisite:
\[
y_i \leq y_j \qquad \forall (i,j) \in Q
\]

6. Bundle bonus activation:
\[
w_{ij} \leq y_i, \quad w_{ij} \leq y_j, \quad w_{ij} \geq y_i + y_j - 1 \qquad \forall (i,j) \in B
\]

7. Variable domains:
\[
x_i \in \{0\} \cup [min_i, max_i] \cap \mathbb{Z} \qquad \forall i \in I
\]
\[
y_i, z_c, w_{ij} \in \{0,1\}
\]

Data Mapping:
- file_5_view_0: authorized item options, $I$, $u_i$, $f_i$, $cat(i)$, $min_i$, $max_i$
- file_2_view_0: category constraints, $C$, $min_c$, $max_c$, $F_c$
- file_1_view_0: resource capacity ledger, $R$, $cap_r$ (sum by resource, convert units)
- file_8_view_0: item resource usage, $a_{ir}$ (convert units)
- file_0_view_0: bundle bonus, $B$, $b_{ij}$
- file_4_view_0: incompatible pairs, $P$
- file_7_view_0: prerequisites, $Q$

All indices, parameters, and constraints are defined directly from the supplied tables and their columns. All units are converted as specified. Only authorized items are included. All constraints and bonuses are enforced as described. The objective is the maximum net benefit in USD cents.