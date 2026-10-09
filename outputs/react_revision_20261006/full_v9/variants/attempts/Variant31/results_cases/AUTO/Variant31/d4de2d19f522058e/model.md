## Symbolic Mathematical Model

**Sets**
- $I$: set of item offers (from file_5_view_0, column item_ref)
- $C$: set of categories (from file_2_view_0, column category)
- $R$: set of resources (from file_1_view_0, column resource)
- $B$: set of bundle bonus pairs (from file_0_view_0, columns item_a, item_b)
- $P$: set of incompatible item pairs (from file_4_view_0, columns item_a, item_b)
- $D$: set of requires dependencies (from file_7_view_0, columns item_ref, prerequisite_ref)

**Parameters**
- $u_i$: unit_benefit_cents for item $i$ (file_5_view_0)
- $f_i$: item_fee_cents for item $i$ (file_5_view_0)
- $a_i$: authorized flag for item $i$ (file_5_view_0)
- $l_i$: minimum_lot for item $i$ (file_5_view_0)
- $m_i$: maximum_order for item $i$ (file_5_view_0)
- $cat_i$: category of item $i$ (file_5_view_0)
- $q^{\min}_c$: minimum_quantity for category $c$ (file_2_view_0)
- $q^{\max}_c$: maximum_quantity for category $c$ (file_2_view_0)
- $F_c$: activation_fee_cents for category $c$ (file_2_view_0)
- $v_{ir}$: usage amount of resource $r$ per unit of item $i$ (file_8_view_0; if missing, treat as 0)
- $K_r$: total available capacity for resource $r$ (sum of opening and reservation entries in file_1_view_0)
- $b_{ij}$: bundle bonus_cents for $(i,j)\in B$ (file_0_view_0)
- $P$: set of incompatible pairs $(i,j)$ (file_4_view_0)
- $D$: set of requires dependencies $(i,p)$ (file_7_view_0; $i$ requires $p$)

**Decision Variables**
- $x_i \in \mathbb{Z}_{\geq 0}$: quantity of item $i$ to order
- $y_i \in \{0,1\}$: 1 if $x_i > 0$, 0 otherwise (item selection indicator)
- $z_c \in \{0,1\}$: 1 if any item in category $c$ is selected, 0 otherwise (category activation indicator)
- $w_{ij} \in \{0,1\}$: 1 if both $x_i > 0$ and $x_j > 0$ for bundle $(i,j)\in B$, 0 otherwise

**Objective**
Maximize net benefit in USD cents:
\[
\max \left\{
\sum_{i\in I} u_i x_i
- \sum_{i\in I} f_i y_i
- \sum_{c\in C} F_c z_c
+ \sum_{(i,j)\in B} b_{ij} w_{ij}
\right\}
\]

**Constraints**

1. **Authorization and Lot/Order Bounds**
   - $x_i = 0$ if $a_i = 0$ (unauthorized)
   - $x_i = 0$ or $l_i \leq x_i \leq m_i$ if $a_i = 1$
   - $x_i \in \mathbb{Z}_{\geq 0}$ for all $i\in I$

2. **Item Selection Indicator**
   - $y_i = 1$ if $x_i > 0$, $y_i = 0$ if $x_i = 0$
   - Enforced via: $x_i \leq m_i y_i$, $x_i \geq l_i y_i$ for $a_i=1$; $y_i = 0$ for $a_i=0$

3. **Category Activation Indicator**
   - $z_c = 1$ if $\sum_{i:cat_i=c} x_i > 0$, $z_c = 0$ otherwise
   - Enforced via: $\sum_{i:cat_i=c} x_i \leq M_c z_c$, $\sum_{i:cat_i=c} x_i \geq l^{\min}_c z_c$ where $M_c$ is large enough, $l^{\min}_c$ is the smallest $l_i$ in $c$

4. **Category Quantity Limits**
   - $q^{\min}_c \leq \sum_{i:cat_i=c} x_i \leq q^{\max}_c$ for all $c\in C$

5. **Resource Capacity**
   - $\sum_{i\in I} v_{ir} x_i \leq K_r$ for all $r\in R$

6. **Incompatible Pairs**
   - For each $(i,j)\in P$: $y_i + y_j \leq 1$

7. **Requires Dependencies**
   - For each $(i,p)\in D$: $x_i \leq m_i y_p$

8. **Bundle Bonuses**
   - For each $(i,j)\in B$:
     - $w_{ij} \leq y_i$
     - $w_{ij} \leq y_j$
     - $w_{ij} \geq y_i + y_j - 1$

**Variable Domains**
- $x_i \in \mathbb{Z}_{\geq 0}$, $y_i \in \{0,1\}$, $z_c \in \{0,1\}$, $w_{ij} \in \{0,1\}$

---

## Data Mapping

- **file_5_view_0**: Items $I$, with columns item_ref ($i$), authorized ($a_i$), minimum_lot ($l_i$), maximum_order ($m_i$), category ($cat_i$), unit_benefit_cents ($u_i$), item_fee_cents ($f_i$)
- **file_2_view_0**: Categories $C$, with columns category ($c$), minimum_quantity ($q^{\min}_c$), maximum_quantity ($q^{\max}_c$), activation_fee_cents ($F_c$)
- **file_1_view_0**: Resources $R$, with columns resource ($r$), entry (opening/reservation), amount (sum for $K_r$)
- **file_8_view_0**: Resource usage $v_{ir}$, with columns item_ref ($i$), resource ($r$), amount
- **file_0_view_0**: Bundle bonuses $B$, with columns item_a ($i$), item_b ($j$), bonus_cents ($b_{ij}$)
- **file_4_view_0**: Incompatible pairs $P$, with columns item_a ($i$), item_b ($j$)
- **file_7_view_0**: Requires dependencies $D$, with columns item_ref ($i$), prerequisite_ref ($p$)

---

**All indices, parameters, and constraints are defined directly from the supplied tables and their columns as described above.**