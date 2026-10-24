## Mixed‑Integer Lot‑Sizing Model

**Index set**  
\(T=\{1,2,\dots ,24\}\) ( month \(t\) corresponds to month label \(M0t\) in the data file )

**Parameters** (all taken directly from *monthly_lot_sizing.csv*, table_id `file_0_view_0`)

| Symbol | Source column | Meaning |
|--------|---------------|---------|
| \(d_t\) | `Demand` | demand in month \(t\) |
| \(c_t\) | `ProductionCost` | unit production cost in month \(t\) |
| \(s_t\) | `SetupCost` | fixed setup cost in month \(t\) |
| \(h_t\) | `HoldingCost` | unit inventory‑holding cost in month \(t\) |
| \(p_t\) | `ProductionCapacity` | maximum production quantity in month \(t\) |

**Decision variables**

\[
\begin{aligned}
x_t &\ge 0 &&\text{(units produced in month }t)\\
I_t &\ge 0 &&\text{(ending inventory after month }t)\\
y_t &\in\{0,1\} &&\text{(1 if production is set up in month }t)
\end{aligned}
\qquad \forall t\in T
\]

**Objective**

\[
\min \;\sum_{t\in T}\bigl(c_t\,x_t + s_t\,y_t + h_t\,I_t\bigr)
\]

**Constraints**

1. **Inventory balance**  
   \[
   I_{t-1} + x_t = d_t + I_t \qquad \forall t\in T
   \]
   with the initial condition \(I_0 = 0\).

2. **Production‑capacity / setup link**  
   \[
   x_t \le p_t\,y_t \qquad \forall t\in T
   \]

3. **Final zero inventory**  
   \[
   I_{24} = 0
   \]

4. **Variable domains** (already stated above)

---

## Data Mapping Rules

| Table ID | Column | Parameter |
|----------|--------|-----------|
| `file_0_view_0` | `Demand` | \(d_t\) |
| `file_0_view_0` | `ProductionCost` | \(c_t\) |
| `file_0_view_0` | `SetupCost` | \(s_t\) |
| `file_0_view_0` | `HoldingCost` | \(h_t\) |
| `file_0_view_0` | `ProductionCapacity` | \(p_t\) |

The index set \(T\) is derived from the 24 rows of the table, preserving the month labels (`M01`–`M24`). No additional rows or columns are introduced beyond those explicitly present in the data file.