## Symbolic Mathematical Model

**Sets:**
- $I$: set of authorized items (from file_5_view_0, column item_ref)
- $C$: set of categories (from file_2_view_0, column category)
- $R$: set of resources (from file_1_view_0, column resource)
- $B$: set of bundle pairs (from file_0_view_0, columns item_a, item_b)
- $P$: set of incompatible pairs (from file_4_view_0, columns item_a, item_b)
- $Q$: set of prerequisite pairs (from file_7_view_0, columns item_ref, prerequisite_ref)

**Parameters:**
- $u_i$: unit_benefit_cents for item $i$ (file_5_view_0)
- $f_i$: item_fee_cents for item $i$ (file_5_view_0)
- $cat(i)$: category of item $i$ (file_5_view_0)
- $min_i$, $max_i$: minimum_lot, maximum_order for item $i$ (file_5_view_0)
- $min_c$, $max_c$: minimum_quantity, maximum_quantity for category $c$ (file_2_view_0)
- $F_c$: activation_fee_cents for category $c$ (file_2_view_0)
- $a_{ir}$: per-unit usage of resource $r$ by item $i$ (file_8_view_0, amount, converted to base units)
- $U_r$: total available capacity for resource $r$ (sum of file_1_view_0, amount, converted to base units)
- $b_{ij}$: bonus_cents for bundle $(i,j)$ (file_0_view_0)
- $P$: set of incompatible pairs $(i,j)$ (file_4_view_0)
- $Q$: set of prerequisite pairs $(i,k)$ (file_7_view_0)

**Decision Variables:**
- $x_i \in \mathbb{Z}_+$: number of cases of item $i$ to order ($x_i = 0$ or $x_i \in [min_i, max_i]$)
- $y_i \in \{0,1\}$: 1 if $x_i > 0$, 0 otherwise (item activation)
- $z_c \in \{0,1\}$: 1 if any item in category $c$ is ordered, 0 otherwise (category activation)
- $w_{ij} \in \{0,1\}$: 1 if both $x_i > 0$ and $x_j > 0$ for bundle $(i,j)$, 0 otherwise

**Objective:**
\[
\max \left\{
\sum_{i \in I} u_i x_i
- \sum_{i \in I} f_i y_i
- \sum_{c \in C} F_c z_c
+ \sum_{(i,j) \in B} b_{ij} w_{ij}
\right\}
\]

**Subject to:**

1. **Item order bounds:**
   \[
   x_i = 0 \quad \text{or} \quad min_i \leq x_i \leq max_i \qquad \forall i \in I
   \]
   (Enforced via $x_i \geq min_i y_i$, $x_i \leq max_i y_i$, $x_i \geq 0$, $y_i \in \{0,1\}$)

2. **Category quantity bounds:**
   \[
   min_c \leq \sum_{i \in I: cat(i)=c} x_i \leq max_c \qquad \forall c \in C
   \]

3. **Category activation:**
   \[
   z_c \geq y_i \qquad \forall c \in C, \forall i \in I: cat(i)=c
   \]

4. **Resource capacity:**
   \[
   \sum_{i \in I} a_{ir} x_i \leq U_r \qquad \forall r \in R
   \]
   (Convert all $a_{ir}$ and $U_r$ to base units: 1 liter = 1000 ml, 1 kwh = 1000 wh, 1 hour = 60 minutes.)

5. **Incompatibility:**
   \[
   y_i + y_j \leq 1 \qquad \forall (i,j) \in P
   \]

6. **Prerequisite:**
   \[
   y_i \leq y_k \qquad \forall (i,k) \in Q
   \]

7. **Bundle bonus activation:**
   \[
   w_{ij} \leq y_i, \quad w_{ij} \leq y_j, \quad w_{ij} \geq y_i + y_j - 1 \qquad \forall (i,j) \in B
   \]

8. **Item activation:**
   \[
   y_i = \begin{cases}
   1 & \text{if } x_i \geq min_i \\
   0 & \text{if } x_i = 0
   \end{cases}
   \qquad \forall i \in I
   \]
   (Enforced via $x_i \geq min_i y_i$, $x_i \leq max_i y_i$, $x_i \geq 0$, $y_i \in \{0,1\}$)

9. **Variable domains:**
   \[
   x_i \in \mathbb{Z}_+, \quad y_i \in \{0,1\}, \quad z_c \in \{0,1\}, \quad w_{ij} \in \{0,1\}
   \]

---

## Data Mapping

- **file_5_view_0**: Items $I$, with columns item_ref ($i$), category ($cat(i)$), minimum_lot ($min_i$), maximum_order ($max_i$), unit_benefit_cents ($u_i$), item_fee_cents ($f_i$), authorized=1.
- **file_2_view_0**: Categories $C$, with columns category ($c$), minimum_quantity ($min_c$), maximum_quantity ($max_c$), activation_fee_cents ($F_c$).
- **file_8_view_0**: Item resource usage $a_{ir}$, with columns item_ref ($i$), resource ($r$), amount (convert to base units), unit.
- **file_1_view_0**: Resource capacity ledger $U_r$, with columns resource ($r$), amount (sum by resource, convert to base units), unit.
- **file_0_view_0**: Bundle bonuses $B$, with columns item_a ($i$), item_b ($j$), bonus_cents ($b_{ij}$).
- **file_4_view_0**: Incompatible pairs $P$, with columns item_a ($i$), item_b ($j$).
- **file_7_view_0**: Prerequisite pairs $Q$, with columns item_ref ($i$), prerequisite_ref ($k$).

---

**All indices, parameters, and constraints are mapped directly from the supplied tables as described above.**