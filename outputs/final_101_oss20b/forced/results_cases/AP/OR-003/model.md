**Mathematical Model**

Let  

- \(S\) be the set of suppliers (indices \(s\)).  
- \(C\) be the set of customer groups (indices \(c\)).  

Parameters  

- \( \text{cap}_s \) – daily supply capacity of supplier \(s\).  
- \( \text{dem}_c \) – daily demand of customer \(c\).  
- \( \text{cost}_{s,c} \) – transportation cost per unit from supplier \(s\) to customer \(c\).  

Decision variables  

- \( x_{s,c} \ge 0 \) – units shipped from supplier \(s\) to customer \(c\).  

\[
\begin{aligned}
\min\;& \sum_{s\in S}\sum_{c\in C} \text{cost}_{s,c}\,x_{s,c} \\[4pt]
\text{s.t.}\;& \sum_{c\in C} x_{s,c} \;\le\; \text{cap}_s \qquad &&\forall s\in S \\[4pt]
& \sum_{s\in S} x_{s,c} \;=\; \text{dem}_c \qquad &&\forall c\in C \\[4pt]
& x_{s,c}\;\ge\;0 \qquad &&\forall s\in S,\;c\in C
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | ID Column | Value Column | Notes |
|-----------|--------------|-----------|--------------|-------|
| \( \text{cap}_s \) | `file_1_view_0` | `Unnamed: 0` | `supply_capacity` | Supplier identifier |
| \( \text{dem}_c \) | `file_0_view_0` | `customer` | `demand` | Customer identifier |
| \( \text{cost}_{s,c} \) | `file_2_view_0` | `Unnamed: 0` (rows) | `C1`–`C10` (columns) | Supplier‑to‑customer cost matrix |

The index sets \(S\) and \(C\) are the distinct values appearing in the respective ID columns of the source tables. No additional constraints or parameters are introduced beyond those required by the problem statement.