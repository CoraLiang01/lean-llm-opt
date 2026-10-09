## Mathematical Model

**Sets**  
Let $T = \{1,2,\ldots,12\}$ index the months, corresponding to the rows of monthly_lot_sizing.csv.

**Parameters** (from table_id: file_0_view_0, columns as named)
- $d_t$: Demand in month $t$ (Demand)
- $c_t$: Unit production cost in month $t$ (ProductionCost)
- $f_t$: Setup cost in month $t$ (SetupCost)
- $h_t$: Unit holding cost in month $t$ (HoldingCost)
- $C_t$: Production capacity in month $t$ (ProductionCapacity)

**Decision Variables**
- $x_t \geq 0$: Production quantity in month $t$ (continuous, units as in data)
- $inv_t \geq 0$: Inventory at end of month $t$ (continuous)
- $y_t \in \{0,1\}$: 1 if production is set up in month $t$, 0 otherwise (binary)

**Objective**
\[
\min \sum_{t=1}^{12} \left[ c_t x_t + f_t y_t + h_t\, inv_t \right]
\]

**Subject to**

_Inventory balance (for all $t=1$):_
\[
inv_1 = x_1 - d_1
\]

_Inventory balance (for all $t=2,\ldots,12$):_
\[
inv_t = inv_{t-1} + x_t - d_t \qquad \forall t=2,\ldots,12
\]

_Production capacity and setup linking (for all $t=1,\ldots,12$):_
\[
x_t \leq C_t\, y_t
\]

_Initial inventory:_
\[
inv_0 = 0
\]

_No backlogging (for all $t=1,\ldots,12$):_
\[
inv_t \geq 0
\]
\[
x_t \geq 0
\]

_Zero ending inventory:_
\[
inv_{12} = 0
\]

_Setup binary:_
\[
y_t \in \{0,1\} \qquad \forall t=1,\ldots,12
\]

---

### Data Mapping

- $T$: All rows in monthly_lot_sizing.csv, column "Month"
- $d_t$: column "Demand", table_id: file_0_view_0
- $c_t$: column "ProductionCost", table_id: file_0_view_0
- $f_t$: column "SetupCost", table_id: file_0_view_0
- $h_t$: column "HoldingCost", table_id: file_0_view_0
- $C_t$: column "ProductionCapacity", table_id: file_0_view_0

All variables and constraints are indexed over $T$ as defined above.