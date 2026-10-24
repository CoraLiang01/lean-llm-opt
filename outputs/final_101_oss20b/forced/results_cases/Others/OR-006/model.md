## Mathematical Model

**Index sets**  
\(W=\{S1,\dots ,S10\}\) (warehouses)  
\(C=\{C1,\dots ,C10\}\) (customers)

**Parameters**  
\[
\begin{aligned}
d_c &\;=\;\text{daily demand of customer }c \quad (c\in C)\\
s_w &\;=\;\text{daily supply capacity of warehouse }w \quad (w\in W)\\
c_{w,c} &\;=\;\text{transportation cost per unit from }w\text{ to }c \quad (w\in W,\;c\in C)
\end{aligned}
\]

**Decision variables**  
\[
x_{w,c}\;\ge 0 \qquad (w\in W,\;c\in C)
\]
(Quantity shipped from warehouse \(w\) to customer \(c\))

**Objective**  
\[
\min \;\sum_{w\in W}\sum_{c\in C} c_{w,c}\,x_{w,c}
\]

**Constraints**

1. **Demand satisfaction**  
\[
\sum_{w\in W} x_{w,c}\;\ge\; d_c \qquad \forall\,c\in C
\]

2. **Supply capacity**  
\[
\sum_{c\in C} x_{w,c}\;\le\; s_w \qquad \forall\,w\in W
\]

3. **Non‑negativity**  
\[
x_{w,c}\;\ge\;0 \qquad \forall\,w\in W,\;c\in C
\]

---

## Data Mapping

| Parameter | Source Table | Source Columns | Mapping Rule |
|-----------|--------------|----------------|--------------|
| \(d_c\) | `file_0_view_0` | `customer`, `demand` | For each record, set \(d_{\text{customer}} = \text{demand}\). |
| \(s_w\) | `file_1_view_0` | `Unnamed: 0`, `supply_capacity` | For each record, set \(s_{\text{Unnamed: 0}} = \text{supply_capacity}\). |
| \(c_{w,c}\) | `file_2_view_0` | `Unnamed: 0` (row), `C1`–`C10` (columns) | For each row \(w\) and column \(c\), set \(c_{w,c} = \text{value}\). |

*Index sets \(W\) and \(C\) are the distinct values of the `Unnamed: 0` column in `file_1_view_0` and the `customer` column in `file_0_view_0`, respectively.*

All parameters are treated as real numbers; decision variables \(x_{w,c}\) are continuous and non‑negative. The model is a linear programming problem that minimizes total transportation cost while meeting all customer demands and respecting warehouse supply limits.