## Mixed-Integer Lot-Sizing Model

**Sets**  
Let $\mathcal{T} = \{1,2,\ldots,24\}$ index the months, corresponding to the rows of monthly_lot_sizing.csv in order.

**Parameters** (from monthly_lot_sizing.csv, table_id: file_0_view_0)
- $d_t$: Demand in month $t$ (column: Demand)
- $c_t$: Unit production cost in month $t$ (column: ProductionCost)
- $f_t$: Fixed setup cost in month $t$ (column: SetupCost)
- $h_t$: Unit inventory holding cost in month $t$ (column: HoldingCost)
- $K_t$: Production capacity in month $t$ (column: ProductionCapacity)

**Decision Variables**
- $x_t \geq 0$: Production quantity in month $t$
- $I_t \geq 0$: Inventory at end of month $t$
- $y_t \in \{0,1\}$: 1 if production is set up in month $t$, 0 otherwise

**Objective**
\[
\min \sum_{t \in \mathcal{T}} \left( c_t x_t + f_t y_t + h_t I_t \right)
\]

**Constraints**
1. **Inventory Balance (for $t=1$):**
   \[
   x_1 - d_1 = I_1
   \]
   (Initial inventory is zero.)

2. **Inventory Balance (for $t=2,\ldots,24$):**
   \[
   I_{t-1} + x_t - d_t = I_t \qquad \forall t = 2,\ldots,24
   \]

3. **Production Capacity and Setup Linking:**
   \[
   x_t \leq K_t y_t \qquad \forall t \in \mathcal{T}
   \]

4. **Final Inventory Zero:**
   \[
   I_{24} = 0
   \]

5. **Nonnegativity and Binary:**
   \[
   x_t \geq 0,\quad I_t \geq 0,\quad y_t \in \{0,1\} \qquad \forall t \in \mathcal{T}
   \]

---

**Data Mapping**  
- $d_t$, $c_t$, $f_t$, $h_t$, $K_t$ are taken from columns Demand, ProductionCost, SetupCost, HoldingCost, ProductionCapacity, respectively, for each row $t$ (row $t-1$ in file_0_view_0).
- $\mathcal{T}$ is the set of all 24 months, in order as listed in the file.

**All constraints, variables, and parameters are as defined above, with no omitted months or data.**