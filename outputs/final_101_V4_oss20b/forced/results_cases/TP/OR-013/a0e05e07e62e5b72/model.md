##### Decision Variables

$x_i\in\mathbb{Z}_{\geq0}$: number of units of product $i$ that the retailer plans to fulfill, for each $i\in I$.

##### Objective Function

$\displaystyle \max \sum_{i\in I} r_i\,x_i$

##### Constraints

1. Demand limits: $\displaystyle x_i \leq d_i,\quad\forall i\in I$  
2. Inventory limits: $\displaystyle x_i \leq s_i,\quad\forall i\in I$  
3. Non‑negativity and integrality: $\displaystyle x_i\in\mathbb{Z}_{\geq0},\quad\forall i\in I$

$ I=\{\text{4U\_Service\_22},\text{4U\_Service\_36},\text{4U\_Service\_7}\}$  

$ d_{\text{4U\_Service\_22}}=5,\; d_{\text{4U\_Service\_36}}=3,\; d_{\text{4U\_Service\_7}}=3$  

$ s_{\text{4U\_Service\_22}}=30,\; s_{\text{4U\_Service\_36}}=20,\; s_{\text{4U\_Service\_7}}=20$  

$ r_{\text{4U\_Service\_22}}=56.0,\; r_{\text{4U\_Service\_36}}=21.6,\; r_{\text{4U\_Service\_7}}=62.5$  

###### Retrieved Information

```json
{
  "products": [
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
  "initial_inventory": {
    "4U_Service_22": 30,
    "4U_Service_36": 20,
    "4U_Service_7": 20
  }
}
```