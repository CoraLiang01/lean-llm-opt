##### Decision Variables

$x_i\geq0$: number of units of ZZ product $i$ that the store plans to fulfill, for each $i\in I$ (continuous).

##### Objective Function

$\max\sum_{i\in I}r_i\,x_i$

##### Constraints

1. Inventory: $\displaystyle x_i\leq I_i,\quad\forall i\in I$  
2. Demand: $\displaystyle x_i\leq D_i,\quad\forall i\in I$  
3. Non‑negativity: $\displaystyle x_i\geq0,\quad\forall i\in I$

$I=\{\text{ZZ2AO},\text{ZZDW7},\text{ZZM1A},\text{ZZNC5},\text{ZZX6K}\}$  

$D_i$ denotes the demand for product $i$, $I_i$ denotes the initial inventory for product $i$, and $r_i$ denotes the revenue per unit for product $i$.

###### Retrieved Information

```json
{
  "products": [
    "ZZ2AO",
    "ZZDW7",
    "ZZM1A",
    "ZZNC5",
    "ZZX6K"
  ],
  "revenue": {
    "ZZ2AO": 24.38,
    "ZZDW7": 30.12,
    "ZZM1A": 19.52,
    "ZZNC5": 10.79,
    "ZZX6K": 111.81
  },
  "demand": {
    "ZZ2AO": 2,
    "ZZDW7": 4,
    "ZZM1A": 82,
    "ZZNC5": 2,
    "ZZX6K": 2
  },
  "initial_inventory": {
    "ZZ2AO": 10.0,
    "ZZDW7": 20.0,
    "ZZM1A": 530.0,
    "ZZNC5": 10.0,
    "ZZX6K": 10.0
  }
}
```