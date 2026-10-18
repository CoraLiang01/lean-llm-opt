##### Abstract Model

**Index Set**

$ \displaystyle I \;=\; \{\, i \mid \text{product } i \text{ is classified under } \text{‘id999’ in } \text{OnlineRetailSalesDataset.csv}\,\} $

**Parameters**

$ \displaystyle w_i \;=\; \text{Revenue of product } i \quad (i \in I) $

$ \displaystyle d_i \;=\; \text{Demand of product } i \quad (i \in I) $

$ \displaystyle c_i \;=\; \text{Initial Inventory of product } i \quad (i \in I) $

**Variables**

$ \displaystyle x_i \;\in\; \mathbb{Z}, \quad x_i \;\ge\; 0 \quad (i \in I) $

**Objective Function**

$ \displaystyle \max \;\sum_{i \in I} w_i \cdot x_i $

**Constraints**

###### 1. Inventory Constraints

$ \displaystyle x_i \;\le\; c_i , \quad \forall i \in I $

###### 2. Demand Constraints

$ \displaystyle x_i \;\le\; d_i , \quad \forall i \in I $

**Data Mapping**

| Table ID | Column Name | Parameter |
|----------|-------------|-----------|
| file_0_view_0 | Revenue | $w_i$ |
| file_0_view_0 | Demand | $d_i$ |
| file_0_view_0 | Initial Inventory | $c_i$ |

*The dataset “OnlineRetailSalesDataset.csv” contains all products classified under ‘id999’. No additional constraints or dispatch capacity limits are imposed.*