**Mathematical Model**

Let  

* \(W\) – set of warehouse identifiers (rows of *file_0_view_0*).  
* \(C\) – set of customer identifiers (columns of *file_0_view_0*).  

Parameters  

\[
\begin{aligned}
\text{FixedCost}_w &\;=\; \text{Fixed\_Cost of warehouse }w \\
\text{Capacity}_w  &\;=\; \text{Capacity of warehouse }w \\
\text{Demand}_c    &\;=\; \text{Demand of customer }c \\
\text{Cost}_{w,c}  &\;=\; \text{transportation cost per unit from }w\text{ to }c
\end{aligned}
\]

Decision variables  

\[
\begin{aligned}
y_w &\in \{0,1\} &&\text{(1 if warehouse }w\text{ is opened)}\\
x_{w,c} &\in \mathbb{Z}_{\ge 0} &&\text{(units shipped from }w\text{ to }c)
\end{aligned}
\]

---

### Objective  

\[
\min \;\sum_{w\in W}\text{FixedCost}_w\,y_w
      \;+\;\sum_{w\in W}\sum_{c\in C}\text{Cost}_{w,c}\,x_{w,c}
\]

### Constraints  

1. **Demand satisfaction** – each customer’s demand must be met exactly  

\[
\sum_{w\in W} x_{w,c} \;=\; \text{Demand}_c
\qquad \forall\,c\in C
\]

2. **Warehouse capacity** – shipments from a warehouse cannot exceed its capacity when it is opened  

\[
\sum_{c\in C} x_{w,c} \;\le\; \text{Capacity}_w \, y_w
\qquad \forall\,w\in W
\]

3. **Variable domains**  

\[
x_{w,c}\;\ge\;0,\; x_{w,c}\in\mathbb{Z}\quad\forall\,w\in W,\;c\in C
\]
\[
y_w\in\{0,1\}\quad\forall\,w\in W
\]

---

## Data Mapping

| Parameter | Table ID | Row ID column | Column ID column | Notes |
|-----------|----------|---------------|------------------|-------|
| `FixedCost_w` | `file_1_view_0` | `Warehouse ID` | `Fixed_Cost` | Fixed annual opening cost |
| `Capacity_w`  | `file_1_view_0` | `Warehouse ID` | `Capacity` | Maximum service capacity |
| `Demand_c`    | `file_2_view_0` | `Customer ID` | `Demand` | Customer demand |
| `Cost_{w,c}`  | `file_0_view_0` | `Warehouse ID` | `C1`–`C20` | Transportation cost per unit; column name `Ck` corresponds to customer `Ck` |

*All index sets (`W`, `C`) are taken from the row and column identifiers of the respective tables.*