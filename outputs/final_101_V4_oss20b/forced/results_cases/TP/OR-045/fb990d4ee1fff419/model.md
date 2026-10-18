##### Decision Variables

$x_i\in\mathbb{Z}_{\geq0}$: number of units of produce type $i$ ordered daily, for each $i\in I$.

##### Objective Function

$\max\sum_{i\in I}v_i\,x_i$

##### Constraints

1. Capacity: $\sum_{i\in I}w_i\,x_i\leq C$.
2. Non‑negativity and integrality: $x_i\in\mathbb{Z}_{\geq0},\quad\forall i\in I$.

$I=\{\text{Spinach},\text{Shiitake Mushrooms},\text{Apples},\text{Carrots},\text{Basil},\text{Potatoes},\text{Green Beans},\text{Blueberries},\text{Oranges},\text{Watermelons}\}$.

$C=1035$.

$w_{\text{Spinach}}=282,\;v_{\text{Spinach}}=49$  
$w_{\text{Shiitake Mushrooms}}=83,\;v_{\text{Shiitake Mushrooms}}=30$  
$w_{\text{Apples}}=251,\;v_{\text{Apples}}=30$  
$w_{\text{Carrots}}=257,\;v_{\text{Carrots}}=18$  
$w_{\text{Basil}}=88,\;v_{\text{Basil}}=54$  
$w_{\text{Potatoes}}=52,\;v_{\text{Potatoes}}=27$  
$w_{\text{Green Beans}}=198,\;v_{\text{Green Beans}}=91$  
$w_{\text{Blueberries}}=203,\;v_{\text{Blueberries}}=88$  
$w_{\text{Oranges}}=87,\;v_{\text{Oranges}}=78$  
$w_{\text{Watermelons}}=265,\;v_{\text{Watermelons}}=22$  

###### Retrieved Information

{
  "capacity": 1035,
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