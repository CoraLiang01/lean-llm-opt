## Sets
- Let \( S = \{ \text{S1}, \text{S2}, \ldots, \text{S11} \} \) be the set of Walmart stores, as indexed by `Unnamed: 0` in `file_1_view_0` and `file_2_view_0`.
- Let \( C = \{ \text{C1}, \text{C2}, \ldots, \text{C12} \} \) be the set of customer groups, as indexed by `customer` in `file_0_view_0` and columns in `file_2_view_0`.

## Parameters (Data Mapping)
- \( d_c \): Demand of customer group \( c \).
  - Data: `file_0_view_0`, column `demand`, indexed by `customer`.
- \( u_s \): Supply capacity of store \( s \).
  - Data: `file_1_view_0`, column `supply_capacity`, indexed by `Unnamed: 0`.
- \( c_{s,c} \): Transportation cost per unit from store \( s \) to customer group \( c \).
  - Data: `file_2_view_0`, value at row `Unnamed: 0` = \( s \), column \( c \).

## Decision Variables
- \( x_{s,c} \geq 0 \): Quantity transported from store \( s \) to customer group \( c \). (Continuous, non-negative)

## Mathematical Model

\[
\begin{align*}
\text{Minimize} \quad & \sum_{s \in S} \sum_{c \in C} c_{s,c} \cdot x_{s,c} \\[2ex]
\text{subject to} \quad
& \sum_{s \in S} x_{s,c} = d_c \quad && \forall c \in C \\
& \sum_{c \in C} x_{s,c} \leq u_s \quad && \forall s \in S \\
& x_{s,c} \geq 0 \quad && \forall s \in S,\, c \in C
\end{align*}
\]

## Data Mapping

- \( S \): All `Unnamed: 0` in `file_1_view_0` and `file_2_view_0` (S1, S2, ..., S11)
- \( C \): All `customer` in `file_0_view_0` and columns (except `Unnamed: 0`) in `file_2_view_0` (C1, C2, ..., C12)
- \( d_c \): `file_0_view_0`, column `demand`, indexed by `customer`
- \( u_s \): `file_1_view_0`, column `supply_capacity`, indexed by `Unnamed: 0`
- \( c_{s,c} \): `file_2_view_0`, value at row `Unnamed: 0` = \( s \), column \( c \)

## Variable Domains

- \( x_{s,c} \geq 0 \), continuous, for all \( s \in S, c \in C \)

---

**This model ensures all customer demands are met, no store exceeds its supply capacity, and total transportation cost is minimized, using only the provided data.**