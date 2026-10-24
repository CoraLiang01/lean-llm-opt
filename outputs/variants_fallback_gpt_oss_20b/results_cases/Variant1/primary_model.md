## Mixed‑Integer Lot‑Sizing Model

**Index set**  
\(T=\{t\mid t=\text{Month }M01,\dots ,M24\}\)

**Parameters** (all taken from *file_0_view_0*)

| Symbol | Meaning | Source column |
|--------|---------|---------------|
| \(d_t\) | Demand in month \(t\) | `Demand` |
| \(c_t\) | Unit production cost in month \(t\) | `ProductionCost` |
| \(s_t\) | Fixed setup cost in month \(t\) | `SetupCost` |
| \(h_t\) | Unit inventory‑holding cost in month \(t\) | `HoldingCost` |
| \(p_t\) | Production capacity in month \(t\) | `ProductionCapacity` |

**Decision variables**

\[
\begin{aligned}
x_t &\ge 0 &&\text{(units produced in month }t)\\
I_t &\ge 0 &&\text{(ending inventory after month }t)\\
y_t &\in\{0,1\} &&\text{(setup indicator in month }t)
\end{aligned}
\qquad \forall t\in T
\]

**Objective**

\[
\min \;\sum_{t\in T}\Bigl(c_t\,x_t + s_t\,y_t + h_t\,I_t\Bigr)
\]

**Constraints**

1. **Initial inventory**

\[
I_{0}=0
\]

2. **Inventory balance (for each month \(t\))**

\[
I_{t-1}+x_t = d_t + I_t \qquad \forall t\in T
\]

3. **Production‑capacity / setup linking**

\[
x_t \le p_t\,y_t \qquad \forall t\in T
\]

4. **Final zero inventory**

\[
I_{24}=0
\]

5. **Variable domains**

\[
x_t\ge 0,\; I_t\ge 0,\; y_t\in\{0,1\}\qquad \forall t\in T
\]

---

## Data Mapping Rules

| Parameter | Table ID | Column name |
|-----------|----------|-------------|
| \(d_t\) | `file_0_view_0` | `Demand` |
| \(c_t\) | `file_0_view_0` | `ProductionCost` |
| \(s_t\) | `file_0_view_0` | `SetupCost` |
| \(h_t\) | `file_0_view_0` | `HoldingCost` |
| \(p_t\) | `file_0_view_0` | `ProductionCapacity` |

The index set \(T\) is derived from the 24 rows of *monthly_lot_sizing.csv*; each row corresponds to a distinct month \(t\). No additional rows or columns are introduced beyond those explicitly listed above.