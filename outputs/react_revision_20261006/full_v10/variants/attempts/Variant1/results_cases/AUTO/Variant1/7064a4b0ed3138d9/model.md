## Mixed-Integer Lot-Sizing Model

**Sets**
- $T = \{1,2,\ldots,24\}$ : months, indexed by $t$

**Parameters** (from Data Mapping)
- $d_t$ = Demand in month $t$ (column "Demand", table_id: file_0_view_0)
- $c_t$ = Unit production cost in month $t$ (column "ProductionCost", table_id: file_0_view_0)
- $f_t$ = Fixed setup cost in month $t$ (column "SetupCost", table_id: file_0_view_0)
- $h_t$ = Unit inventory holding cost in month $t$ (column "HoldingCost", table_id: file_0_view_0)
- $u_t$ = Production capacity in month $t$ (column "ProductionCapacity", table_id: file_0_view_0)

**Decision Variables**
- $x_t \geq 0$ : production quantity in month $t$
- $I_t \geq 0$ : ending inventory after month $t$
- $y_t \in \{0,1\}$ : 1 if production is set up in month $t$, 0 otherwise

**Objective**
Minimize total cost:
$$
\min \sum_{t=1}^{24} \left[ c_t x_t + f_t y_t + h_t I_t \right]
$$

**Constraints**
1. **Inventory Balance (for $t=1$):**
$$
x_1 - d_1 = I_1
$$

2. **Inventory Balance (for $t=2,\ldots,24$):**
$$
I_{t-1} + x_t - d_t = I_t \qquad \forall t=2,\ldots,24
$$

3. **Initial Inventory:**
$$
I_0 = 0
$$

4. **Final Inventory:**
$$
I_{24} = 0
$$

5. **Production Capacity and Setup Linking:**
$$
x_t \leq u_t y_t \qquad \forall t=1,\ldots,24
$$

6. **Nonnegativity and Binary Restrictions:**
$$
x_t \geq 0,\quad I_t \geq 0,\quad y_t \in \{0,1\} \qquad \forall t=1,\ldots,24
$$

**Data Mapping**
- $d_t$ = "Demand" in row $t-1$ of table_id: file_0_view_0
- $c_t$ = "ProductionCost" in row $t-1$ of table_id: file_0_view_0
- $f_t$ = "SetupCost" in row $t-1$ of table_id: file_0_view_0
- $h_t$ = "HoldingCost" in row $t-1$ of table_id: file_0_view_0
- $u_t$ = "ProductionCapacity" in row $t-1$ of table_id: file_0_view_0

**Index correspondence:** $t=1$ corresponds to "M01", $t=2$ to "M02", ..., $t=24$ to "M24" in the "Month" column.

**All parameters are sourced from table_id: file_0_view_0, columns as specified above.**