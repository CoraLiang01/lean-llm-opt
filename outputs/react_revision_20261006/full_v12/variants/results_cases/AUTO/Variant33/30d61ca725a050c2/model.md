## Mathematical Model

**Sets**
- $I$: set of authorized item options (from item options tables, where authorized = 1)
- $G$: set of categories (from category limits table)
- $R$: set of resources (from resource usage and capacity ledger tables)
- $B$: set of bundle pairs (from bundle bonuses table)
- $C$: set of incompatible pairs (from incompatible pairs table)
- $Q$: set of prerequisite pairs (from requires table)

**Parameters**
- $b_i$: per-unit benefit of item $i$ (sum of amount_cents for $i$ in benefit components table)
- $f_i$: item fixed fee (item_fee) for $i$ (from item fixed fees table)
- $g$: category
- $g(i)$: category of item $i$ (from item options tables)
- $F_g$: category activation fee for $g$ (from category limits table)
- $L_g, U_g$: min/max quantity for category $g$ (from category limits table)
- $l_i, u_i$: minimum_lot and maximum_order for $i$ (from item options tables)
- $a_{ir}$: usage of resource $r$ per unit of $i$ (from item resource usage tables; missing = 0)
- $S_r$: total available of resource $r$ (sum of capacity_ledger entries for $r$)
- $f_{ij}$: bundle bonus for $(i,j)\in B$ (from bundle bonuses table)
- $C$: set of unordered incompatible pairs $(i,j)$ (from incompatible pairs table)
- $Q$: set of prerequisite pairs $(i,p)$ (from requires table; $i$ requires $p$)

**Decision Variables**
- $x_i \in \{0\} \cup \{l_i, l_i+1, ..., u_i\}$: integer quantity of item $i$ selected
- $y_i \in \{0,1\}$: 1 if $x_i > 0$, 0 otherwise
- $z_g \in \{0,1\}$: 1 if any $x_i > 0$ for $i$ in category $g$, 0 otherwise
- $w_{ij} \in \{0,1\}$: 1 if both $x_i > 0$ and $x_j > 0$ for $(i,j)\in B$, 0 otherwise

**Objective**
\[
\max \left\{
\sum_{i\in I} b_i x_i
- \sum_{i\in I} f_i y_i
- \sum_{g\in G} F_g z_g
+ \sum_{(i,j)\in B} f_{ij} w_{ij}
\right\}
\]

**Constraints**

1. **Authorization and bounds**
   - $x_i = 0$ if $i$ is unauthorized (i.e., not in $I$)
   - $x_i \in \{0\} \cup \{l_i, ..., u_i\}$ for $i\in I$

2. **Item activation**
   - $y_i = 1$ if $x_i > 0$, $y_i = 0$ if $x_i = 0$ for $i\in I$
   - $x_i \leq u_i y_i$ for $i\in I$
   - $x_i \geq l_i y_i$ for $i\in I$

3. **Category activation**
   - $z_g \geq y_i$ for all $i\in I$ with $g(i)=g$, $g\in G$
   - $z_g \leq \sum_{i: g(i)=g} y_i$ for $g\in G$ (or $z_g = \max_{i: g(i)=g} y_i$)

4. **Category quantity limits**
   - $L_g \leq \sum_{i: g(i)=g} x_i \leq U_g$ for $g\in G$

5. **Resource constraints**
   - $\sum_{i\in I} a_{ir} x_i \leq S_r$ for all $r\in R$

6. **Incompatibility**
   - $y_i + y_j \leq 1$ for all $(i,j)\in C$

7. **Requires**
   - $y_i \leq y_p$ for all $(i,p)\in Q$

8. **Bundle bonuses**
   - $w_{ij} \leq y_i$, $w_{ij} \leq y_j$, $w_{ij} \geq y_i + y_j - 1$ for all $(i,j)\in B$

**Variable domains**
- $x_i \in \{0\} \cup \{l_i, ..., u_i\}$, integer, for $i\in I$
- $y_i \in \{0,1\}$ for $i\in I$
- $z_g \in \{0,1\}$ for $g\in G$
- $w_{ij} \in \{0,1\}$ for $(i,j)\in B$

---

## Data Mapping

- file_0_view_0: benefit components ($b_i$)
- file_1_view_0: bundle bonuses ($B$, $f_{ij}$)
- file_2_view_0: resource capacity ledger ($S_r$)
- file_3_view_0: category limits and fees ($G$, $L_g$, $U_g$, $F_g$)
- file_4_view_0: item identity (for display only)
- file_5_view_0: incompatible pairs ($C$)
- file_6_view_0, file_7_view_0: item options ($I$, $g(i)$, $l_i$, $u_i$, authorization)
- file_8_view_0: item fixed fees ($f_i$)
- file_10_view_0: requires ($Q$)
- file_11_view_0, file_12_view_0: item resource usage ($a_{ir}$)

All indices, parameters, and constraints are defined directly from these tables. All units are in cents and matching base units as provided.