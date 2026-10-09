## Symbolic Mathematical Model

**Sets**
- $I$: set of authorized items (from file_2_view_0, column item_ref)
- $C$: set of categories (from file_4_view_0, column category)
- $R$: set of resources (from file_3_view_0, column resource)
- $B$: set of bundle pairs (from file_7_view_0, columns item_a, item_b)
- $P$: set of prerequisite pairs (from file_0_view_0, columns item_ref, prerequisite_ref)
- $Q$: set of incompatible pairs (from file_5_view_0, columns item_a, item_b)

**Parameters**
- $u_i$: unit_benefit_cents for item $i$ (file_2_view_0)
- $f_i$: item_fee_cents for item $i$ (file_2_view_0)
- $cat(i)$: category of item $i$ (file_2_view_0)
- $min_i$: minimum_lot for item $i$ (file_2_view_0)
- $max_i$: maximum_order for item $i$ (file_2_view_0)
- $a_{i,r}$: per-unit usage of resource $r$ by item $i$ (file_3_view_0, amount; 0 if missing)
- $u_{i,r}$: unit for $a_{i,r}$ (file_3_view_0, unit)
- $cap_r$: total available capacity for resource $r$ (sum of file_8_view_0, amount, for each $r$)
- $capunit_r$: unit for $cap_r$ (file_8_view_0, unit)
- $min_c$, $max_c$, $F_c$: minimum_quantity, maximum_quantity, activation_fee_cents for category $c$ (file_4_view_0)
- $bonus_{ij}$: bonus_cents for bundle $(i,j)$ (file_7_view_0)
- $prereq(i)$: set of prerequisite items for $i$ (file_0_view_0)
- $inc(i)$: set of items incompatible with $i$ (file_5_view_0)

**Unit conversions**
- For $u_{i,r}$ and $capunit_r$:
    - If $u_{i,r}$ = "liter", $a_{i,r} \gets 1000 \cdot a_{i,r}$ (ml)
    - If $u_{i,r}$ = "hour", $a_{i,r} \gets 60 \cdot a_{i,r}$ (minute)
    - If $u_{i,r}$ = "kwh", $a_{i,r} \gets 1000 \cdot a_{i,r}$ (wh)
    - If $u_{i,r}$ = "ml", no change
    - If $u_{i,r}$ = "minute", no change
    - If $u_{i,r}$ = "wh", no change

**Decision Variables**
- $x_i \in \mathbb{Z}_+$: quantity of item $i$ to order ($x_i = 0$ or $min_i \leq x_i \leq max_i$)
- $y_i \in \{0,1\}$: 1 if $x_i > 0$, 0 otherwise
- $z_c \in \{0,1\}$: 1 if any $x_i > 0$ for $i$ with $cat(i)=c$, 0 otherwise
- $b_{ij} \in \{0,1\}$: 1 if both $x_i > 0$ and $x_j > 0$ for bundle $(i,j)\in B$, 0 otherwise

**Objective**
Maximize net benefit in USD cents:
\[
\max \left(
\sum_{i \in I} u_i x_i
- \sum_{i \in I} f_i y_i
- \sum_{c \in C} F_c z_c
+ \sum_{(i,j)\in B} bonus_{ij} b_{ij}
\right)
\]

**Constraints**

1. **Item quantity bounds**
   \[
   x_i = 0 \quad \text{or} \quad min_i \leq x_i \leq max_i \qquad \forall i \in I
   \]
   \[
   y_i = \begin{cases}
   1 & \text{if } x_i > 0 \\
   0 & \text{if } x_i = 0
   \end{cases} \qquad \forall i \in I
   \]
   (Enforced via: $x_i \leq max_i y_i$, $x_i \geq min_i y_i$, $x_i \geq 0$, $y_i \in \{0,1\}$)

2. **Category activation**
   \[
   z_c \geq y_i \qquad \forall i \in I,\, cat(i)=c
   \]
   \[
   z_c \in \{0,1\} \qquad \forall c \in C
   \]

3. **Category quantity bounds**
   \[
   min_c \leq \sum_{i:cat(i)=c} x_i \leq max_c \qquad \forall c \in C
   \]

4. **Resource constraints**
   \[
   \sum_{i \in I} a_{i,r} x_i \leq cap_r \qquad \forall r \in R
   \]
   (with all $a_{i,r}$ and $cap_r$ in common units as above)

5. **Incompatibility**
   \[
   y_i + y_j \leq 1 \qquad \forall (i,j) \in Q
   \]

6. **Prerequisites**
   \[
   y_i \leq y_{prereq} \qquad \forall (i,prereq) \in P
   \]

7. **Bundle bonuses**
   \[
   b_{ij} \leq y_i,\quad b_{ij} \leq y_j,\quad b_{ij} \geq y_i + y_j - 1 \qquad \forall (i,j) \in B
   \]
   (and $b_{ij}=0$ if either $i$ or $j$ is unauthorized)

**Variable domains**
\[
x_i \in \{0\} \cup [min_i, max_i] \cap \mathbb{Z} \qquad \forall i \in I
\]
\[
y_i, z_c, b_{ij} \in \{0,1\}
\]

---

## Data Mapping

- **Items**: $I$ = all item_ref with authorized=1 from file_2_view_0
- **Categories**: $C$ = all category from file_4_view_0
- **Resources**: $R$ = all resource from file_3_view_0 and file_8_view_0
- **Bundle pairs**: $B$ = all (item_a, item_b) from file_7_view_0
- **Prerequisite pairs**: $P$ = all (item_ref, prerequisite_ref) from file_0_view_0
- **Incompatible pairs**: $Q$ = all (item_a, item_b) from file_5_view_0
- **Unit conversions**: as described above, using file_3_view_0 and file_8_view_0 units
- **Resource capacities**: $cap_r$ = sum of amount for each resource $r$ in file_8_view_0
- **All parameters**: as described, from the corresponding columns in the respective tables

---

**Report the maximum net benefit in USD cents.**