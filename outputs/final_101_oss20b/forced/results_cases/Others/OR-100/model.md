## Mathematical Model

**Sets**  
\(W=\{\text{Casting},\text{Milling},\text{Finishing},\text{Assembly},\text{QA \& Packaging}\}\)  
\(C=\{C1,C2,\dots ,C111\}\)

**Parameters**  

\[
\begin{aligned}
t_{w,c} &\;=\;\text{processing time per unit of component }c\text{ in workshop }w
          &&\text{(from file_0_view_0)}\\
p_{c}   &\;=\;\text{unit price of component }c
          &&\text{(from file_1_view_0)}\\
H_{w}   &\;=\;\text{total available working hours in workshop }w
          &&\text{(from file_2_view_0, only for workshops with a record)}
\end{aligned}
\]

**Decision Variables**  

\[
x_{c}\;\ge 0 \qquad \forall c\in C
\]

**Objective**  

\[
\max \;\sum_{c\in C} p_{c}\,x_{c}
\]

**Constraints**  

For every workshop \(w\) that has a recorded capacity \(H_{w}\):

\[
\sum_{c\in C} t_{w,c}\,x_{c}\;\le\; H_{w}\qquad \forall w\in W\text{ with }H_{w}\text{ defined}
\]

(If a workshop such as *Finishing* has no entry in `total_working_hours.csv`, no capacity constraint is imposed for that workshop.)

---

## Data Mapping

| Parameter | Source Table | Key Column(s) | Value Column(s) | Mapping Rule |
|-----------|--------------|---------------|-----------------|--------------|
| \(t_{w,c}\) | `file_0_view_0` | `Unnamed: 0` (workshop) → `w`<br>`C1`–`C111` (component) → `c` | `C1`–`C111` (time values) | For each row \(w\) and each column \(c\), assign the numeric value to \(t_{w,c}\). |
| \(p_{c}\) | `file_1_view_0` | `Unnamed: 0` (component) → `c` | `unit_price` | For each record, assign the numeric value to \(p_{c}\). |
| \(H_{w}\) | `file_2_view_0` | `workshop` → `w` | `total_hours` | For each record, assign the numeric value to \(H_{w}\). Only workshops present in this table receive a capacity constraint. |

All indices \(w\) and \(c\) are taken from the full set of rows/columns in the respective source tables; no additional filtering is applied unless explicitly stated in the query. The model uses continuous, non‑negative variables \(x_{c}\) and linear constraints, thus it is a linear programming problem.