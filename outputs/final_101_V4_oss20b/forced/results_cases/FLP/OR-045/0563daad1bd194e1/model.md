##### Decision Variables

$x_i \in \mathbb{Z}_{\geq 0}$: number of units of produce type $i$ ordered daily, for $i \in I$.

##### Objective Function

$\displaystyle \max \sum_{i\in I} v_i\,x_i$

##### Constraints

1. Capacity constraint: $\displaystyle \sum_{i\in I} w_i\,x_i \;\leq\; C$.

2. Non‑negativity and integrality: $\displaystyle x_i \;\geq\; 0,\;\; x_i \in \mathbb{Z},\;\; \forall i\in I$.

##### Parameters

$I=\{1,2,3,4,5,6,7,8,9,10\}$  

$C=1035$  

$w=\bigl[282,\;83,\;251,\;257,\;88,\;52,\;198,\;203,\;87,\;265\bigr]$  

$v=\bigl[49,\;30,\;30,\;18,\;54,\;27,\;91,\;88,\;78,\;22\bigr]$  

$\,\,$(The index $i$ corresponds to the following product names in order:  
1. Spinach, 2. Shiitake Mushrooms, 3. Apples, 4. Carrots, 5. Basil, 6. Potatoes, 7. Green Beans, 8. Blueberries, 9. Oranges, 10. Watermelons.)

###### Retrieved Information

```json
{
  "capacity": {
    "Capacity": 1035
  },
  "products": [
    {"ProductName":"Spinach","Weight":282,"Value":49},
    {"ProductName":"Shiitake Mushrooms","Weight":83,"Value":30},
    {"ProductName":"Apples","Weight":251,"Value":30},
    {"ProductName":"Carrots","Weight":257,"Value":18},
    {"ProductName":"Basil","Weight":88,"Value":54},
    {"ProductName":"Potatoes","Weight":52,"Value":27},
    {"ProductName":"Green Beans","Weight":198,"Value":91},
    {"ProductName":"Blueberries","Weight":203,"Value":88},
    {"ProductName":"Oranges","Weight":87,"Value":78},
    {"ProductName":"Watermelons","Weight":265,"Value":22}
  ]
}
```