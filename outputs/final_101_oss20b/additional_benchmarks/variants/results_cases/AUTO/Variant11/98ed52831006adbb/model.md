## Mathematical Model

Let  
\(T=\{1,2,\dots ,12\}\) be the set of months (month 1 = M01, … , month 12 = M12).

**Parameters** (all taken from *monthly_lot_sizing.csv*)  

\[
\begin{aligned}
&D_t &&\text{Demand in month }t,\\
&c_t &&\text{Unit production cost in month }t,\\
&s_t &&\text{Setup cost in month }t,\\
&h_t &&\text{Unit holding cost in month }t,\\
&C_t &&\text{Production capacity in month }t.
\end{aligned}
\]

**Decision variables**

\[
\begin{aligned}
x_t &\ge 0 &&\text{Production quantity in month }t,\\
\text{inv}_t &\ge 0 &&\text{Ending inventory after month }t,\\
y_t &\in\{0,1\} &&\text{Setup indicator in month }t.
\end{aligned}
\]

**Objective**

\[
\min \;\sum_{t\in T}\bigl(c_t\,x_t + s_t\,y_t + h_t\,\text{inv}_t\bigr)
\]

**Constraints**

1. **Inventory balance** (initial inventory \( \text{inv}_0 = 0\))  

   \[
   \text{inv}_{t-1} + x_t = D_t + \text{inv}_t \qquad \forall\,t\in T
   \]

2. **Capacity–setup link**  

   \[
   x_t \le C_t\,y_t \qquad \forall\,t\in T
   \]

3. **Zero ending inventory**  

   \[
   \text{inv}_{12}=0
   \]

4. **Variable domains**  

   \[
   x_t,\;\text{inv}_t \ge 0,\qquad y_t\in\{0,1\}\quad \forall\,t\in T
   \]

---

## Data Mapping

| Index set | Definition |
|-----------|------------|
| \(T\) | \(\{1,\dots,12\}\) (months M01–M12) |

| Parameter | Source column | Source table |
|-----------|---------------|--------------|
| \(D_t\) | `Demand` | `file_0_view_0` |
| \(c_t\) | `ProductionCost` | `file_0_view_0` |
| \(s_t\) | `SetupCost` | `file_0_view_0` |
| \(h_t\) | `HoldingCost` | `file_0_view_0` |
| \(C_t\) | `ProductionCapacity` | `file_0_view_0` |

| Variable | Domain |
|----------|--------|
| \(x_t\) | \(\ge 0\) |
| \(\text{inv}_t\) | \(\ge 0\) |
| \(y_t\) | \(\{0,1\}\) |

All constraints and the objective use only the parameters and variables defined above. No additional data or assumptions are introduced.