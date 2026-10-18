##### Index Sets  
- \(W\): Set of warehouses, indexed by \(w\).  
- \(S\): Set of stores, indexed by \(s\).  

##### Parameters  
- \(d_s\): Daily demand of store \(s\) (from **file_0_view_0**, column *demand*).  
- \(c_w\): Daily supply capacity of warehouse \(w\) (from **file_1_view_0**, column *supply_capacity*).  
- \(t_{ws}\): Transportation cost per unit from warehouse \(w\) to store \(s\) (from **file_2_view_0**, columns *Unnamed: 0* for warehouse id and *C1*–*C10* for store ids).  

##### Decision Variables  
- \(x_{ws}\): Quantity of product shipped from warehouse \(w\) to store \(s\) (integer, \(x_{ws}\in\mathbb{Z}_{\ge0}\)).  

##### Objective Function  
$$
\min \sum_{w\in W}\sum_{s\in S} t_{ws}\,x_{ws}
$$  

##### Constraints  

###### 1. Demand Satisfaction  
$$
\sum_{w\in W} x_{ws}=d_s,\qquad \forall s\in S
$$  

###### 2. Supply Capacity  
$$
\sum_{s\in S} x_{ws}\le c_w,\qquad \forall w\in W
$$  

###### 3. Variable Domains  
$$
x_{ws}\in\mathbb{Z}_{\ge0},\qquad \forall w\in W,\;\forall s\in S
$$  

##### Data Mapping  

| Parameter | Source Table | Source Columns |
|-----------|--------------|----------------|
| \(d_s\) | **file_0_view_0** | *customer*, *demand* |
| \(c_w\) | **file_1_view_0** | *Unnamed: 0*, *supply_capacity* |
| \(t_{ws}\) | **file_2_view_0** | *Unnamed: 0* (warehouse id), *C1*–*C10* (store ids) |

---