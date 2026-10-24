**Mathematical Model**

Let  

* \(I\) – set of all product records in *RetailSalesDataset.csv*  
* \(i \in I\) – index of a product  

Parameters (all read directly from the CSV file)

| Symbol | Meaning | Source column |
|--------|---------|---------------|
| \(r_i\) | Revenue per unit of product \(i\) | `Revenue` |
| \(d_i\) | Total deterministic demand for product \(i\) | `Demand` |
| \(s_i\) | Initial inventory of product \(i\) | `Initial Inventory` |

Decision variables  

\[
x_i \in \mathbb{Z}_{\ge 0}\qquad \forall i\in I
\]

*\(x_i\)* = number of units of product \(i\) the store chooses to sell.

**Objective**

\[
\max \;\; \sum_{i\in I} r_i\,x_i
\]

**Constraints**

1. **Demand limits**  
   \[
   x_i \;\le\; d_i \qquad \forall i\in I
   \]

2. **Inventory limits**  
   \[
   x_i \;\le\; s_i \qquad \forall i\in I
   \]

No other constraints are imposed.

---

### Data Mapping

| Parameter | CSV file path | CSV column name |
|-----------|---------------|-----------------|
| \(r_i\) | `Test_Dataset/Large-scale-or/NRM_testing/NRM7/RetailSalesDataset.csv` | `Revenue` |
| \(d_i\) | `Test_Dataset/Large-scale-or/NRM_testing/NRM7/RetailSalesDataset.csv` | `Demand` |
| \(s_i\) | `Test_Dataset/Large-scale-or/NRM_testing/NRM7/RetailSalesDataset.csv` | `Initial Inventory` |

The index set \(I\) is implicitly defined by the rows of the CSV file; each row corresponds to a distinct product. No numerical values or row counts are included in the model.