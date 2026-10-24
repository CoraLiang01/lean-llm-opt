**Mathematical Model**

Let  

* \(I=\{S1,S2,\dots ,S7\}\) – set of warehouses  
* \(J=\{C1,C2,\dots ,C7\}\) – set of customers (musicians/bands)

Parameters  

\[
\begin{aligned}
f_i &\quad\text{fixed operating cost of warehouse }i\\
d_j &\quad\text{demand of customer }j\\
t_{ij}&\quad\text{transportation cost per unit from warehouse }i\text{ to customer }j
\end{aligned}
\]

Decision variables  

\[
\begin{aligned}
y_i &\in\{0,1\}\quad\text{(warehouse }i\text{ is activated)}\\
x_{ij} &\in\mathbb{Z}_{\ge 0}\quad\text{(units shipped from }i\text{ to }j)
\end{aligned}
\]

---

### Objective  

\[
\min \;\; \sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} t_{ij}\,x_{ij}
\]

---

### Constraints  

1. **Demand satisfaction**  
   \[
   \sum_{i\in I} x_{ij} \;\ge\; d_j \qquad \forall j\in J
   \]

2. **Warehouse activation (big‑M)**  
   \[
   x_{ij} \;\le\; M\,y_i \qquad \forall i\in I,\;\forall j\in J
   \]
   where \(M=\sum_{j\in J} d_j\) (a sufficiently large constant).

3. **Variable domains**  
   \[
   y_i \in \{0,1\}\quad\forall i\in I,\qquad
   x_{ij}\in\mathbb{Z}_{\ge 0}\quad\forall i\in I,\;\forall j\in J
   \]

---

## Data Mapping

| Parameter | Table ID | Key Column | Value Column |
|-----------|----------|------------|--------------|
| \(f_i\) | `file_1_view_0` | `Unnamed: 0` | `fixed_costs` |
| \(d_j\) | `file_0_view_0` | `customer` | `demand` |
| \(t_{ij}\) | `file_2_view_0` | `Unnamed: 0` | `C1`, `C2`, `C3`, `C4` (only defined for these customers; missing entries are treated as zero) |

*Indices*  
- Warehouse index \(i\) is taken from the `Unnamed: 0` column of `file_1_view_0` and `file_2_view_0`.  
- Customer index \(j\) is taken from the `customer` column of `file_0_view_0` and the column names `C1`–`C4` of `file_2_view_0`.  

The model uses all returned entities; for customers \(C5\)–\(C7\) the transportation cost parameter \(t_{ij}\) is implicitly zero because no data are provided.