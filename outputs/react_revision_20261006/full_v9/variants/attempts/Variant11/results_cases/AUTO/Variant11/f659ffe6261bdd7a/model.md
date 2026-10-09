## Mathematical Model

**Sets**  
Let $\mathcal{T} = \{1,2,\ldots,12\}$ index the months, corresponding to rows M01–M12 in the data.

**Parameters** (from monthly_lot_sizing.csv, table_id: file_0_view_0)  
For each $t \in \mathcal{T}$:
- $d_t$: Demand in month $t$ (column "Demand")
- $c_t$: Unit production cost in month $t$ (column "ProductionCost")
- $s_t$: Setup cost in month $t$ (column "SetupCost")
- $h_t$: Unit holding cost from month $t$ to $t+1$ (column "HoldingCost")
- $u_t$: Production capacity in month $t$ (column "ProductionCapacity")

**Decision Variables**  
For each $t \in \mathcal{T}$:
- $x_t \geq 0$: Production quantity in month $t$
- $inv_t \geq 0$: Inventory at end of month $t$
- $y_t \in \{0,1\}$: 1 if production occurs in month $t$, 0 otherwise

**Objective**  
Minimize total cost:
$$
\min \sum_{t=1}^{12} \left[ c_t x_t + s_t y_t + h_t\, inv_t \right]
$$

**Constraints**

1. **Inventory Balance (for all $t \in \mathcal{T}$):**
   $$
   inv_t = inv_{t-1} + x_t - d_t \qquad \forall t=1,\ldots,12
   $$
   with $inv_0 = 0$ (initial inventory is zero).

2. **Production Capacity and Setup Linking (for all $t \in \mathcal{T}$):**
   $$
   x_t \leq u_t\, y_t \qquad \forall t=1,\ldots,12
   $$

3. **No Backlogging (for all $t \in \mathcal{T}$):**
   $$
   inv_t \geq 0 \qquad \forall t=1,\ldots,12
   $$

4. **Zero Ending Inventory:**
   $$
   inv_{12} = 0
   $$

5. **Nonnegativity and Binary Restrictions:**
   $$
   x_t \geq 0,\quad y_t \in \{0,1\} \qquad \forall t=1,\ldots,12
   $$

**Data Mapping**  
- $\mathcal{T}$: All rows in file_0_view_0, column "Month" (M01–M12)
- $d_t$: file_0_view_0, column "Demand"
- $c_t$: file_0_view_0, column "ProductionCost"
- $s_t$: file_0_view_0, column "SetupCost"
- $h_t$: file_0_view_0, column "HoldingCost"
- $u_t$: file_0_view_0, column "ProductionCapacity"