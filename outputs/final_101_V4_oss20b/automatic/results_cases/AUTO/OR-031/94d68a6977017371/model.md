##### Index Sets

- $I$ : set of dairy products.

##### Parameters

- $r_i$ : revenue per unit of product $i$.
- $d_i$ : deterministic demand for product $i$ over the sales horizon.
- $c_i$ : initial inventory available for product $i$.

##### Variables

- $x_i \in \mathbb{Z}_{\ge 0}$ : number of units of product $i$ to fulfill.

##### Objective Function

$$\max \sum_{i \in I} r_i \, x_i$$

##### Constraints

###### 1. Demand Constraints

$$x_i \le d_i, \quad \forall i \in I$$

###### 2. Inventory Constraints

$$x_i \le c_i, \quad \forall i \in I$$

###### 3. Variable Domain

$$x_i \in \mathbb{Z}_{\ge 0}, \quad \forall i \in I$$

##### Data Mapping

| Symbol | Source Table ID | Source Column |
|--------|-----------------|---------------|
| $r_i$  | file_0_view_0   | Revenue       |
| $d_i$  | file_0_view_0   | Demand        |
| $c_i$  | file_0_view_0   | Initial Inventory |
| $I$    | file_0_view_0   | Full_Product_Name (each distinct product name) |