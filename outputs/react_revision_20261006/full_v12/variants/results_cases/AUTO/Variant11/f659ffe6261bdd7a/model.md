## Mathematical Model

**Sets**  
$T = \{1,2,\ldots,12\}$ (months, ordered as in table_id file_0_view_0)

**Parameters** (from table_id file_0_view_0)  
For each $t \in T$:
- $d_t$ = Demand in month $t$ (column: Demand)
- $c_t$ = Unit production cost in month $t$ (column: ProductionCost)
- $f_t$ = Setup cost in month $t$ (column: SetupCost)
- $h_t$ = Unit holding cost in month $t$ (column: HoldingCost)
- $u_t$ = Production capacity in month $t$ (column: ProductionCapacity)

**Variables**  
For each $t \in T$:
- $x_t \geq 0$ (production quantity in month $t$)
- $inv_t \geq 0$ (ending inventory after month $t$)
- $y_t \in \{0,1\}$ (1 if production occurs in month $t$, 0 otherwise)

**Objective**  
Minimize total cost:
$$
\min \sum_{t=1}^{12} \left[ c_t x_t + f_t y_t + h_t\, inv_t \right]
$$

**Constraints**

1. **Inventory balance** (for $t=1$):
$$
x_1 = d_1 + inv_1
$$

(for $t=2,\ldots,12$):
$$
inv_{t-1} + x_t = d_t + inv_t \qquad \forall t=2,\ldots,12
$$

2. **Initial inventory**
$$
inv_0 = 0
$$

3. **Ending inventory**
$$
inv_{12} = 0
$$

4. **Production capacity and setup linking** (for all $t$):
$$
x_t \leq u_t\, y_t \qquad \forall t \in T
$$

5. **Nonnegativity and binary**
$$
x_t \geq 0,\quad inv_t \geq 0,\quad y_t \in \{0,1\} \qquad \forall t \in T
$$

---

### Data Mapping

- $T$: All rows in table_id file_0_view_0, ordered by "Month"
- $d_t$: "Demand" column, file_0_view_0
- $c_t$: "ProductionCost" column, file_0_view_0
- $f_t$: "SetupCost" column, file_0_view_0
- $h_t$: "HoldingCost" column, file_0_view_0
- $u_t$: "ProductionCapacity" column, file_0_view_0

All parameters are mapped directly from the corresponding columns in table_id file_0_view_0.