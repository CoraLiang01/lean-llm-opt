## Mathematical Model

**Index set**  
\(T=\{1,2,\dots ,12\}\)  (months \(M01,\dots ,M12\))

**Parameters** (all taken from *file_0_view_0*)

| Symbol | Source column | Meaning |
|--------|---------------|---------|
| \(D_t\) | `Demand` | Monthly demand |
| \(c_t\) | `ProductionCost` | Unit production cost |
| \(s_t\) | `SetupCost` | Fixed setup cost |
| \(h_t\) | `HoldingCost` | Unit holding cost |
| \(C_t\) | `ProductionCapacity` | Production capacity |

**Decision variables**

| Symbol | Domain | Meaning |
|--------|--------|---------|
| \(x_t\) | \(\mathbb{R}_+\) | Production quantity in month \(t\) |
| \(\text{inv}_t\) | \(\mathbb{R}_+\) | Ending inventory after month \(t\) |
| \(y_t\) | \(\{0,1\}\) | 1 if production occurs in month \(t\), 0 otherwise |

**Objective**

\[
\min \;\sum_{t\in T}\bigl(c_t\,x_t + s_t\,y_t + h_t\,\text{inv}_t\bigr)
\]

**Constraints**

1. **Inventory balance**  
   \[
   \text{inv}_{t-1} + x_t = D_t + \text{inv}_t \qquad \forall\,t\in T
   \]
   with \(\text{inv}_0 = 0\).

2. **Capacity–setup linkage**  
   \[
   x_t \le C_t\,y_t \qquad \forall\,t\in T
   \]

3. **Zero ending inventory**  
   \[
   \text{inv}_{12}=0
   \]

4. **Non‑negativity**  
   \[
   x_t \ge 0,\;\text{inv}_t \ge 0 \qquad \forall\,t\in T
   \]

5. **Binary setup**  
   \[
   y_t \in \{0,1\} \qquad \forall\,t\in T
   \]

---

## Data Mapping

| Index | Table | Column | Mapping |
|-------|-------|--------|---------|
| \(t\) | `file_0_view_0` | `Month` | \(t=1\) for `M01`, …, \(t=12\) for `M12` |
| \(D_t\) | `file_0_view_0` | `Demand` | \(D_t\) value for month \(t\) |
| \(c_t\) | `file_0_view_0` | `ProductionCost` | \(c_t\) value for month \(t\) |
| \(s_t\) | `file_0_view_0` | `SetupCost` | \(s_t\) value for month \(t\) |
| \(h_t\) | `file_0_view_0` | `HoldingCost` | \(h_t\) value for month \(t\) |
| \(C_t\) | `file_0_view_0` | `ProductionCapacity` | \(C_t\) value for month \(t\) |

All parameters are taken directly from the corresponding columns of the single
table *file_0_view_0*; no additional data or assumptions are introduced.