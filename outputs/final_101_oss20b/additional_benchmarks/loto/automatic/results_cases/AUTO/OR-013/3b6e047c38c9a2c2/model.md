##### Objective Function: 

$ \quad \quad \max \sum_i \text{Revenue}_i \cdot x_i $

##### Constraints

###### 1. Demand Constraints: 

$ \quad \quad x_i \leq \text{Demand}_i, \quad \forall i $

###### 2. Inventory Constraints: 

$ \quad \quad x_i \leq \text{Initial Inventory}_i, \quad \forall i $

###### 3. Non-negativity and Integrality Constraints: 

$ \quad \quad x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i $

I indexes the three “4U” products; i indexes Product Name. Revenue_i is the revenue per unit, Demand_i is the maximum demand, and Initial Inventory_i is the available inventory. Variables are nonnegative integers representing the number of units to fulfill.

###### Retrieved Information

{
  "items": [
    "4U_Service_22",
    "4U_Service_36",
    "4U_Service_7"
  ],
  "revenue": {
    "4U_Service_22": 56.0,
    "4U_Service_36": 21.6,
    "4U_Service_7": 62.5
  },
  "demand": {
    "4U_Service_22": 5,
    "4U_Service_36": 3,
    "4U_Service_7": 3
  },
  "inventory": {
    "4U_Service_22": 30,
    "4U_Service_36": 20,
    "4U_Service_7": 20
  }
}