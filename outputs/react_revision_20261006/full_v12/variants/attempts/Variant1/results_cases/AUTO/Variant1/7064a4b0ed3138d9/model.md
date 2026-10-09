## Mixed-Integer Lot-Sizing Model

**Sets:**
- $T = \{1,2,\ldots,24\}$ : months, where $t=1$ corresponds to "M01", $t=2$ to "M02", ..., $t=24$ to "M24".

**Parameters (from monthly_lot_sizing.csv, table_id: file_0_view_0):**
- $d_t$ = Demand in month $t$ ("Demand")
- $c_t$ = Unit production cost in month $t$ ("ProductionCost")
- $f_t$ = Fixed setup cost in month $t$ ("SetupCost")
- $h_t$ = Unit inventory holding cost in month $t$ ("HoldingCost")
- $u_t$ = Production capacity in month $t$ ("ProductionCapacity")

**Decision Variables:**
- $x_t \geq 0$ : production quantity in month $t$
- $I_t \geq 0$ : ending inventory after month $t$
- $y_t \in \{0,1\}$ : 1 if production is set up in month $t$, 0 otherwise

**Objective:**
\[
\min \sum_{t=1}^{24} \left( c_t x_t + f_t y_t + h_t I_t \right)
\]

**Constraints:**

- **Inventory balance (for $t=1$):**
  \[
  x_1 - d_1 = I_1
  \]
  (since initial inventory is zero)

- **Inventory balance (for $t=2,\ldots,24$):**
  \[
  I_{t-1} + x_t - d_t = I_t \qquad \forall t=2,\ldots,24
  \]

- **Production capacity and setup linking (for all $t$):**
  \[
  x_t \leq u_t y_t \qquad \forall t=1,\ldots,24
  \]

- **Final zero-inventory requirement:**
  \[
  I_{24} = 0
  \]

- **Nonnegativity and integrality:**
  \[
  x_t \geq 0,\quad I_t \geq 0,\quad y_t \in \{0,1\} \qquad \forall t=1,\ldots,24
  \]

---

**Data Mapping (all from table_id: file_0_view_0, monthly_lot_sizing.csv):**
- $d_t$ = "Demand", $c_t$ = "ProductionCost", $f_t$ = "SetupCost", $h_t$ = "HoldingCost", $u_t$ = "ProductionCapacity", for $t$ corresponding to "Month" $M01$ to $M24$ (rows 0 to 23).