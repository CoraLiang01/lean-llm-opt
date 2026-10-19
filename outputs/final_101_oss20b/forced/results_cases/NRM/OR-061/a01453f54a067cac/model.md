**Index sets**  
\(I=\{i\mid i\text{ is a supplier}\}\)  
\(J=\{j\mid j\text{ is a branch}\}\)

**Parameters**  
\(\text{fixed\_cost}_i\in\mathbb{R}_{+}\) (from table file_1_view_0, column fixed_costs, supplier id from column Unnamed: 0)  
\(\text{demand}_j\in\mathbb{R}_{+}\) (from table file_0_view_0, column demand, branch id from column customer)  
\(\text{transport\_cost}_{ij}\in\mathbb{R}_{+}\) (from table file_2_view_0, columns C1…C5, supplier id from column Unnamed: 0, branch id from column header)

**Decision variables**  
\(y_i\in\{0,1\}\) (1 if supplier i is opened, 0 otherwise)  
\(x_{ij}\ge 0\) (quantity of goods supplied from supplier i to branch j)

**Objective**  
\[
\min \;\sum_{i\in I}\text{fixed\_cost}_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J}\text{transport\_cost}_{ij}\,x_{ij}
\]

**Constraints**  

1. **Demand satisfaction**  
\[
\sum_{i\in I}x_{ij} \;=\; \text{demand}_j \quad \forall j\in J
\]

2. **Supplier activation**  
\[
x_{ij} \;\le\; \text{demand}_j\,y_i \quad \forall i\in I,\; j\in J
\]

3. **Variable domains**  
\[
y_i\in\{0,1\}\quad \forall i\in I
\]
\[
x_{ij}\ge 0 \quad \forall i\in I,\; j\in J
\]

**Data Mapping**  

| Parameter | Table ID | Column(s) | Source ID column |
|-----------|----------|-----------|------------------|
| \(\text{fixed\_cost}_i\) | file_1_view_0 | fixed_costs | Unnamed: 0 |
| \(\text{demand}_j\) | file_0_view_0 | demand | customer |
| \(\text{transport\_cost}_{ij}\) | file_2_view_0 | C1…C5 | Unnamed: 0 (row), column header (branch) |