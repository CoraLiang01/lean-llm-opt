**Symbolic Model**

Let  

* \(S\) – set of suppliers (indexed by \(i\))  
* \(C\) – set of supermarkets (indexed by \(j\))

Parameters  

\[
\begin{aligned}
f_i &\;=\;\text{fixed cost of opening supplier }i, && i\in S\\
d_j &\;=\;\text{demand of supermarket }j, && j\in C\\
t_{ij}&\;=\;\text{transportation cost per unit from supplier }i\text{ to supermarket }j, && i\in S,\;j\in C
\end{aligned}
\]

Decision variables  

\[
\begin{aligned}
y_i &\in\{0,1\} &&\text{(supplier }i\text{ is open)}\\
x_{ij} &\ge 0 &&\text{(units shipped from }i\text{ to }j)
\end{aligned}
\]

Objective  

\[
\min \;\sum_{i\in S} f_i\,y_i \;+\; \sum_{i\in S}\sum_{j\in C} t_{ij}\,x_{ij}
\]

Constraints  

1. **Demand satisfaction**  
   \[
   \sum_{i\in S} x_{ij}=d_j ,\qquad \forall j\in C
   \]

2. **Supplier activation**  
   \[
   x_{ij}\;\le\; d_j\,y_i ,\qquad \forall i\in S,\;\forall j\in C
   \]

3. **Variable domains**  
   \[
   y_i\in\{0,1\},\; x_{ij}\ge 0
   \]

---

**Data Mapping**

| Parameter | Source Table | Source Column | Identifier Column |
|-----------|--------------|---------------|-------------------|
| \(f_i\) | `file_1_view_0` | `fixed_costs` | `Unnamed: 0` (supplier id) |
| \(d_j\) | `file_0_view_0` | `demand` | `customer` (supermarket id) |
| \(t_{ij}\) | `file_2_view_0` | `C1`–`C25` (one column per supermarket) | `Unnamed: 0` (supplier id) |

All indices \(i\) and \(j\) are taken from the complete sets of suppliers and supermarkets returned in the observation. No additional data or assumptions are introduced beyond what is supplied.