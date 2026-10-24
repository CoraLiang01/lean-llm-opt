## Mathematical Model

**Index sets**  
\(I=\{1,\dots ,8\}\) (Sections)  
\(J=\{1,\dots ,10\}\) (Products)

**Parameters**  
\(\text{Cap}_i\) – display capacity of section \(i\) (from *capacity.csv*, column *Capacity*)  
\(w_j\) – shelf‑space requirement of product \(j\) (from *products.csv*, column *Weight*)  
\(v_j\) – selling price of product \(j\) (from *products.csv*, column *Value*)

**Decision variables**  
\(x_{ij}\in \mathbb{Z}_{+}\) – number of units of product \(j\) placed in section \(i\)

**Objective**  
\[
\max \;\sum_{i\in I}\sum_{j\in J} v_j\,x_{ij}
\]

**Constraints**  
Capacity of each section  
\[
\sum_{j\in J} w_j\,x_{ij}\;\le\;\text{Cap}_i \qquad \forall\,i\in I
\]

Non‑negativity and integrality  
\[
x_{ij}\;\ge\;0,\; x_{ij}\in\mathbb{Z}\qquad \forall\,i\in I,\;j\in J
\]

---

## Data Mapping

| Table ID | Source File | Columns | Parameter Mapping |
|----------|-------------|---------|-------------------|
| file_0_view_0 | capacity.csv | SectionID → \(i\) <br> Capacity → \(\text{Cap}_i\) | \(\text{Cap}_i\) |
| file_1_view_0 | products.csv | ProductName → \(j\) <br> Value → \(v_j\) <br> Weight → \(w_j\) | \(v_j,\; w_j\) |

*All rows in the tables are used; no filtering is applied.*