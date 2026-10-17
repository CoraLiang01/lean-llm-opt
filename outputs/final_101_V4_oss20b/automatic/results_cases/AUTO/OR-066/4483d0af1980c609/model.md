**Mathematical Model**

Let  

* \(I=\{i\}\) be the set of suppliers (identified by the values in column **Unnamed: 0** of *fixed_cost.csv*).  
* \(J=\{j\}\) be the set of supermarkets (identified by the values in column **customer** of *demand.csv*).  

Parameters  

\[
\begin{aligned}
f_i &\;=\;\text{fixed cost of supplier }i \quad &&\text{(from *fixed_cost.csv*)}\\
d_j &\;=\;\text{demand of supermarket }j \quad &&\text{(from *demand.csv*)}\\
c_{ij}&\;=\;\text{unit transportation cost from supplier }i\text{ to supermarket }j
&&\text{(from *transportation_costs.csv*)}\\
M &\;=\;\sum_{j\in J} d_j \quad &&\text{(big‑M upper bound)}
\end{aligned}
\]

Decision variables  

\[
\begin{aligned}
y_i &\in\{0,1\} &&\text{(supplier }i\text{ is activated)}\\
x_{ij} &\ge 0 &&\text{(units shipped from }i\text{ to }j)
\end{aligned}
\]

**Objective**

\[
\min \;\sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}\;+\;\sum_{i\in I} f_i\,y_i
\]

**Constraints**

1. **Demand satisfaction**  
   \[
   \sum_{i\in I} x_{ij}=d_j \quad \forall j\in J
   \]

2. **Supplier activation (big‑M)**  
   \[
   \sum_{j\in J} x_{ij}\;\le\; M\,y_i \quad \forall i\in I
   \]

3. **Variable domains**  
   \[
   x_{ij}\ge 0,\qquad y_i\in\{0,1\}\quad \forall i\in I,\;j\in J
   \]

---

### Data Mapping

| Parameter | Source Table | Row Identifier | Column Identifier |
|-----------|--------------|----------------|-------------------|
| \(f_i\)   | `file_1_view_0` | `Unnamed: 0` | `fixed_costs` |
| \(d_j\)   | `file_0_view_0` | `customer` | `demand` |
| \(c_{ij}\)| `file_2_view_0` | `Unnamed: 0` | `C1`, `C2` (columns correspond to supermarkets) |

*All indices \(i\) and \(j\) are taken from the complete set of identifiers present in the respective source tables.*