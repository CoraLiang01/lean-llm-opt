**Mathematical model**

Let  

* \(I=\{i\mid i\in\text{file\_1\_view\_0.Unnamed: 0}\}\) (11 Walmart stores)  
* \(J=\{j\mid j\in\text{file\_0\_view\_0.customer}\}\) (12 customer groups)

Parameters  

* \(d_j\) = demand of customer \(j\) (from **file_0_view_0.demand**)  
* \(s_i\) = supply capacity of store \(i\) (from **file_1_view_0.supply_capacity**)  
* \(c_{ij}\) = unit transportation cost from store \(i\) to customer \(j\) (from **file_2_view_0**; row axis = Unnamed: 0, column axis = customer)

Decision variables  

\[
x_{ij}\ge 0 \qquad \forall i\in I,\; j\in J
\]

Objective  

\[
\min \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

Constraints  

1. **Demand satisfaction**  
   \[
   \sum_{i\in I} x_{ij} \;\ge\; d_j \qquad \forall j\in J
   \]

2. **Supply capacity**  
   \[
   \sum_{j\in J} x_{ij} \;\le\; s_i \qquad \forall i\in I
   \]

3. **Non‑negativity** (already in variable definition)

---

**Data mapping rules**

| Parameter | Source table | Row axis | Column axis | Value column |
|-----------|--------------|----------|-------------|--------------|
| \(d_j\)   | file_0_view_0 | –        | customer    | demand       |
| \(s_i\)   | file_1_view_0 | –        | Unnamed: 0  | supply_capacity |
| \(c_{ij}\)| file_2_view_0 | Unnamed: 0 | customer | cost value (string → numeric) |

*Index sets*  
- \(I\) is the set of all values in **file_1_view_0.Unnamed: 0**.  
- \(J\) is the set of all values in **file_0_view_0.customer**.

*Matrix shape*  
- \(c_{ij}\) is an 11 × 12 matrix (11 stores × 12 customers) as verified by the validation checks.