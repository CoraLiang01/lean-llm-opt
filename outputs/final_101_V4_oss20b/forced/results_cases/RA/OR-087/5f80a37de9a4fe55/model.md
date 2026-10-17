**Mathematical Model**

Let  

* \(P=\{1,\dots ,80\}\) – product indices (corresponding to columns \(A1\)–\(A80\)).  
* \(D=\{1,\dots ,22\}\) – days of the month.

Decision variables  

\[
q_{i,d}\in \mathbb{Z}_{\ge 0}\qquad\text{(units of product \(i\) produced on day \(d\))}
\]
\[
y_i\in\{0,1\}\qquad\text{(1 if line for product \(i\) is activated, 0 otherwise)}
\]

Parameters (all taken directly from the CSV files)

\[
\begin{aligned}
\text{Demand}_i      &\;=\;\text{value in row 0, column }A_i \text{ of file\_0\_view\_0}\\
\text{Price}_i       &\;=\;\text{value in row 1, column }A_i \text{ of file\_0\_view\_0}\\
\text{Cost}_i        &\;=\;\text{value in row 2, column }A_i \text{ of file\_0\_view\_0}\\
\text{Quota}_i       &\;=\;\text{value in row 3, column }A_i \text{ of file\_0\_view\_0}\\
\text{ActCost}_i     &\;=\;\text{value in row 0, column }A_i \text{ of file\_1\_view\_0}\\
\text{Batch}_i       &\;=\;\text{value in row 0, column }A_i \text{ of file\_2\_view\_0}
\end{aligned}
\]

---

### Objective

\[
\max \;\sum_{i\in P}\sum_{d\in D}\bigl(\text{Price}_i-\text{Cost}_i\bigr)\,q_{i,d}
      \;-\;\sum_{i\in P}\text{ActCost}_i\,y_i
\]

---

### Constraints

1. **Demand satisfaction**

\[
\sum_{d\in D} q_{i,d}\;\le\;\text{Demand}_i \qquad \forall i\in P
\]

2. **Daily production quota (only if line activated)**

\[
q_{i,d}\;\le\;\text{Quota}_i\,y_i \qquad \forall i\in P,\; d\in D
\]

3. **Minimum batch size (only if line activated)**

\[
q_{i,d}\;\ge\;\text{Batch}_i\,y_i \qquad \forall i\in P,\; d\in D
\]

4. **Variable domains**

\[
q_{i,d}\in\mathbb{Z}_{\ge 0}\qquad \forall i\in P,\; d\in D
\]
\[
y_i\in\{0,1\}\qquad \forall i\in P
\]

---

**Data Mapping**

| Parameter | Source Table | Source Row | Column |
|-----------|--------------|------------|--------|
| Demand\_i | file\_0\_view\_0 | 0 | A\_i |
| Price\_i  | file\_0\_view\_0 | 1 | A\_i |
| Cost\_i   | file\_0\_view\_0 | 2 | A\_i |
| Quota\_i  | file\_0\_view\_0 | 3 | A\_i |
| ActCost\_i| file\_1\_view\_0 | 0 | A\_i |
| Batch\_i  | file\_2\_view\_0 | 0 | A\_i |

All indices \(i\) run over the 80 product columns \(A1\)–\(A80\); all days \(d\) run over 1–22. The model uses only the data supplied in the observation; no additional values are invented.