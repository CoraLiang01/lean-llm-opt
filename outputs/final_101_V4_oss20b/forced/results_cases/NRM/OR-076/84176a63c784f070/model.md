**Symbolic Model**

\[
\begin{aligned}
\text{Sets} \quad & W \; \text{: set of warehouses} \\
& C \; \text{: set of customers}
\end{aligned}
\]

\[
\begin{aligned}
\text{Parameters} \quad & f_w \;\; (w\in W) &&\text{fixed opening cost of warehouse }w \\
& \text{cap}_w \;\; (w\in W) &&\text{maximum service capacity of warehouse }w \\
& d_c \;\; (c\in C) &&\text{demand of customer }c \\
& t_{w,c} \;\; (w\in W,\;c\in C) &&\text{transportation cost per unit from }w\text{ to }c
\end{aligned}
\]

\[
\begin{aligned}
\text{Decision variables} \quad & y_w \in \{0,1\} &&\text{(1 if warehouse }w\text{ is opened)} \\
& x_{w,c} \in \mathbb{Z}_{+} &&\text{(units shipped from }w\text{ to }c)
\end{aligned}
\]

\[
\begin{aligned}
\text{Objective:}\quad & \min \;\; \sum_{w\in W} f_w\,y_w \;+\; \sum_{w\in W}\sum_{c\in C} t_{w,c}\,x_{w,c}
\end{aligned}
\]

\[
\begin{aligned}
\text{Constraints:} \\
& \sum_{w\in W} x_{w,c} \;=\; d_c \qquad &&\forall\,c\in C \quad\text{(demand satisfaction)}\\[4pt]
& \sum_{c\in C} x_{w,c} \;\le\; \text{cap}_w\,y_w \qquad &&\forall\,w\in W \quad\text{(capacity)}\\[4pt]
& x_{w,c} \;\ge\; 0,\; x_{w,c}\in\mathbb{Z} \qquad &&\forall\,w\in W,\;c\in C\\
& y_w \in \{0,1\} \qquad &&\forall\,w\in W
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column(s) | Mapping |
|-----------|--------------|------------------|---------|
| \(f_w\) | `file_1_view_0` | `Fixed_Cost` | Row identifier `Warehouse ID` → \(w\) |
| \(\text{cap}_w\) | `file_1_view_0` | `Capacity` | Row identifier `Warehouse ID` → \(w\) |
| \(d_c\) | `file_2_view_0` | `Demand` | Row identifier `Customer ID` → \(c\) |
| \(t_{w,c}\) | `file_0_view_0` | Matrix cells | Row identifier `Warehouse ID` → \(w\); column identifier `C1`–`C20` → \(c\) |

*Sets \(W\) and \(C\) are the collections of all distinct `Warehouse ID` and `Customer ID` values appearing in the respective tables.*