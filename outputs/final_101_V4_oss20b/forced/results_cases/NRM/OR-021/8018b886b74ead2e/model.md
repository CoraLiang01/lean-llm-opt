**Mathematical Model**

```markdown
Sets
-----
I  : set of products

Parameters
----------
Revenue_i   : revenue per unit of product i
Demand_i    : deterministic demand of product i
Inv_i       : initial inventory of product i

Variables
---------
x_i ∈ ℤ_+   : number of units of product i to be fulfilled,  i ∈ I

Objective
---------
$$\max \;\sum_{i\in I} Revenue_i \; x_i$$

Constraints
-----------
$$0 \;\le\; x_i \;\le\; Demand_i \qquad \forall i\in I$$
$$0 \;\le\; x_i \;\le\; Inv_i \qquad \forall i\in I$$
```

**Data Mapping**

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| `I` (index) | `file_0_view_0` | `Product Name` |
| `Revenue_i` | `file_0_view_0` | `Revenue` |
| `Demand_i` | `file_0_view_0` | `Demand` |
| `Inv_i` | `file_0_view_0` | `Initial Inventory` |

All rows of the table `file_0_view_0` are used to instantiate the index set `I` and the corresponding parameter values. No additional data or constraints are introduced beyond those specified in the query.