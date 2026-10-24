**Mathematical Model**

Let  

* \(I=\{SC1,SC2,\dots ,SC10\}\) – candidate service‑centre indices  
* \(J=\{C1,C2,\dots ,C15\}\) – customer indices  

Parameters  

\[
\begin{aligned}
f_i &\quad\text{fixed opening cost of centre }i,\\
c_{ij}&\quad\text{service cost of assigning customer }j\text{ to centre }i .
\end{aligned}
\]

Decision variables  

\[
\begin{aligned}
y_i &\in\{0,1\}\quad\text{(centre }i\text{ is opened)} ,\\
x_{ij}&\in\{0,1\}\quad\text{(customer }j\text{ is assigned to centre }i)} .
\end{aligned}
\]

**Objective**

\[
\min \;\sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

**Constraints**

1. **Customer assignment**  
   \[
   \sum_{i\in I} x_{ij}=1 ,\qquad \forall j\in J
   \]

2. **Capacity of opened centres**  
   \[
   \sum_{j\in J} x_{ij}\;\le\;4\,y_i ,\qquad \forall i\in I
   \]

3. **Variable domains**  
   \[
   y_i\in\{0,1\},\; x_{ij}\in\{0,1\}\quad \forall i\in I,\;j\in J
   \]

---

### Data Mapping

| Parameter | Source | Mapping rule |
|-----------|--------|--------------|
| \(f_i\) | `file_1_view_0` | `Fixed Opening Cost` where `Service Center` = \(i\) |
| \(c_{ij}\) | `file_0_view_0` | value in column `i` where `Customer` = \(j\) |

All other data (index sets, bounds, etc.) are derived directly from the CSV files as shown above. No additional numerical values are enumerated in the model.