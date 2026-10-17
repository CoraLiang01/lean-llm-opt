**Mathematical Model**

Let  

* \(I\) – set of “Baby” products obtained from *EuropeSalesRecords.csv* after applying the filter  
  \[
  I=\{\,i \mid \text{Product Name}_i \text{ contains the substring “Baby”}\,\}.
  \]

* Parameters (indexed by \(i\in I\)):
  * \(\text{rev}_i\) – revenue per unit of product \(i\)  
    \[
    \text{rev}_i = \text{EuropeSalesRecords}[i].\text{Revenue}
    \]
  * \(\text{dem}_i\) – demand quantity of product \(i\)  
    \[
    \text{dem}_i = \text{EuropeSalesRecords}[i].\text{Demand}
    \]
  * \(\text{inv}_i\) – initial inventory of product \(i\)  
    \[
    \text{inv}_i = \text{EuropeSalesRecords}[i].\text{Initial Inventory}
    \]

* Decision variables:
  * \(x_i \ge 0\) – number of units of product \(i\) to be fulfilled (continuous).

**Objective**

\[
\max \;\sum_{i\in I} \text{rev}_i \, x_i
\]

**Constraints**

1. **Demand satisfaction**  
   \[
   x_i \;\le\; \text{dem}_i , \qquad \forall i\in I
   \]

2. **Inventory limitation**  
   \[
   x_i \;\le\; \text{inv}_i , \qquad \forall i\in I
   \]

3. **Non‑negativity**  
   \[
   x_i \;\ge\; 0 , \qquad \forall i\in I
   \]

---

### Data Mapping Rules

| Parameter | Source Table | Column | Mapping Rule |
|-----------|--------------|--------|--------------|
| \(I\) | *EuropeSalesRecords.csv* (view `file_0_view_0`) | `Product Name` | \(I = \{i \mid \text{Product Name}_i \text{ contains “Baby”}\}\) |
| \(\text{rev}_i\) | *EuropeSalesRecords.csv* (view `file_0_view_0`) | `Revenue` | \(\text{rev}_i = \text{EuropeSalesRecords}[i].\text{Revenue}\) |
| \(\text{dem}_i\) | *EuropeSalesRecords.csv* (view `file_0_view_0`) | `Demand` | \(\text{dem}_i = \text{EuropeSalesRecords}[i].\text{Demand}\) |
| \(\text{inv}_i\) | *EuropeSalesRecords.csv* (view `file_0_view_0`) | `Initial Inventory` | \(\text{inv}_i = \text{EuropeSalesRecords}[i].\text{Initial Inventory}\) |

*Table ID:* `file_0_view_0`  
*Columns used:* `Product Name`, `Revenue`, `Demand`, `Initial Inventory`  

No additional parameters or constraints are introduced beyond those derived from the query and the supplied data.