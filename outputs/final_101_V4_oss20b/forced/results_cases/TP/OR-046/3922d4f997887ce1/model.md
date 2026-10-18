##### Decision Variables

$x_i\geq0$: number of units of product $i$ to be ordered each day (continuous).

##### Objective Function

$\max\sum_{i\in I}v_i\,x_i$

##### Constraints

1. Capacity: $\sum_{i\in I}w_i\,x_i\leq C$.

2. Non‑negativity: $x_i\geq0,\quad\forall i\in I$.

$I=\{1,2,\dots ,10\}$ where each index corresponds to a product in the order listed below.  
$C=875$ is the total stock capacity.

###### Retrieved Information

{
  "capacity": {
    "C": 875
  },
  "products": [
    {
      "index": 1,
      "ProductName": "Spinach",
      "Weight": 230,
      "Value": 64
    },
    {
      "index": 2,
      "ProductName": "Shiitake Mushrooms",
      "Weight": 637,
      "Value": 75
    },
    {
      "index": 3,
      "ProductName": "Apples",
      "Weight": 773,
      "Value": 68
    },
    {
      "index": 4,
      "ProductName": "Carrots",
      "Weight": 653,
      "Value": 11
    },
    {
      "index": 5,
      "ProductName": "Basil",
      "Weight": 755,
      "Value": 91
    },
    {
      "index": 6,
      "ProductName": "Potatoes",
      "Weight": 670,
      "Value": 31
    },
    {
      "index": 7,
      "ProductName": "Green Beans",
      "Weight": 505,
      "Value": 90
    },
    {
      "index": 8,
      "ProductName": "Blueberries",
      "Weight": 821,
      "Value": 56
    },
    {
      "index": 9,
      "ProductName": "Oranges",
      "Weight": 83,
      "Value": 10
    },
    {
      "index": 10,
      "ProductName": "Watermelons",
      "Weight": 249,
      "Value": 24
    }
  ]
}