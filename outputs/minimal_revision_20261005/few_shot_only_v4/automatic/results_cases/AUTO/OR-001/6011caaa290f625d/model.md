##### Decision Variables

Let $x_{ij} \geq 0$ be the continuous quantity shipped from distribution center $i \in I$ to customer group $j \in J$.

##### Sets

- $I = \{$S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12, S13, S14, S15, S16, S17, S18$\}$ (distribution centers)
- $J = \{$C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C13, C14, C15, C16, C17, C18$\}$ (customer groups)

##### Parameters

- $d_j$: demand of customer $j \in J$
- $s_i$: supply capacity of distribution center $i \in I$
- $c_{ij}$: transportation cost per unit from $i$ to $j$

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction** (each customer group receives at least its demand):

   \[
   \sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
   \]

2. **Supply capacity** (each distribution center does not exceed its capacity):

   \[
   \sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
   \]

3. **Non-negativity**:

   \[
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   \]

---

#### Data Mapping

- **customer_demand.csv**: $d_j$ for $j \in J$ (column: "customer", "demand")
- **supply_capacity.csv**: $s_i$ for $i \in I$ (column: "Unnamed: 0", "supply_capacity")
- **transportation_costs.csv**: $c_{ij}$ for $i \in I$, $j \in J$ (rows: "Unnamed: 0" = $i$, columns: $j$)

All identifiers and coefficients are to be used exactly as retrieved.