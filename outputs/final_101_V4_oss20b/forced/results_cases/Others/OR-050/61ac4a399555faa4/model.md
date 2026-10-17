## Mathematical Model

**Sets**  
\(I=\{1,\dots ,10\}\) (Shelf identifiers from *capacity.csv*)  
\(J=\{\text{Smartphone},\text{Laptop},\dots ,\text{E‑Reader}\}\) (Product names from *products.csv*)

**Parameters**  
\(\text{Cap}_i\) – capacity of shelf \(i\) (kg) (from *capacity.csv*)  
\(\text{Val}_j\) – value per unit of product \(j\) (yuan) (from *products.csv*)  
\(\text{Wgt}_j\) – weight per unit of product \(j\) (kg) (from *products.csv*)

**Decision Variables**  
\(x_{ij}\in \mathbb{Z}_{+}\) – number of units of product \(j\) placed on shelf \(i\)

**Objective**  
\[
\max \;\sum_{i\in I}\sum_{j\in J}\text{Val}_j\,x_{ij}
\]

**Constraints**

1. **Shelf capacity**  
\[
\sum_{j\in J}\text{Wgt}_j\,x_{ij}\;\le\;\text{Cap}_i
\qquad \forall\,i\in I
\]

2. **Minimum quantity of the first product**  
Let \(j^{*}=\text{Smartphone}\) (the first product in *products.csv*).  
\[
\sum_{i\in I}x_{i\,j^{*}}\;\ge\;5
\]

3. **Non‑negativity and integrality**  
\[
x_{ij}\;\ge\;0,\qquad x_{ij}\in\mathbb{Z}\quad \forall\,i\in I,\;j\in J
\]

---

## Data Mapping

| Parameter | Source Table | Column | Mapping |
|-----------|--------------|--------|---------|
| \(\text{Cap}_i\) | `file_0_view_0` | `Capacity` | \(\text{Cap}_{\text{ShelfID}} = \text{Capacity}\) |
| \(\text{Val}_j\) | `file_1_view_0` | `Value` | \(\text{Val}_{\text{ProductName}} = \text{Value}\) |
| \(\text{Wgt}_j\) | `file_1_view_0` | `Weight` | \(\text{Wgt}_{\text{ProductName}} = \text{Weight}\) |

*Indices*  
- \(i\) ranges over the `ShelfID` values in `file_0_view_0`.  
- \(j\) ranges over the `ProductName` values in `file_1_view_0`.  

All constraints and the objective use only these mapped parameters and the decision variables defined above.