##### Decision Variables

$x_{ij}\geq0$: number of units of product $j\in J$ placed on display $i\in I$ (continuous).

##### Objective Function

$\max\sum_{i\in I}\sum_{j\in J}v_{j}x_{ij}$

##### Constraints

1. Weight capacity: $\sum_{j\in J}w_{j}x_{ij}\leq c_{i},\quad\forall i\in I$.
2. Minimum first product: $\sum_{i\in I}x_{i1}\geq5$.
3. Non-negativity: $x_{ij}\geq0$ continuous.

$I=\{1,2,3,4,5,6,7,8,9,10\}$ and $J=\{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20\}$.

###### Retrieved Information

{
  "shelves": [
    {"ShelfID":"1","Capacity":"5.0"},
    {"ShelfID":"2","Capacity":"7.0"},
    {"ShelfID":"3","Capacity":"6.0"},
    {"ShelfID":"4","Capacity":"8.0"},
    {"ShelfID":"5","Capacity":"5.5"},
    {"ShelfID":"6","Capacity":"9.0"},
    {"ShelfID":"7","Capacity":"6.5"},
    {"ShelfID":"8","Capacity":"7.5"},
    {"ShelfID":"9","Capacity":"8.2"},
    {"ShelfID":"10","Capacity":"5.7"}
  ],
  "products": [
    {"ProductName":"Smartphone","Value":"200","Weight":"1.0"},
    {"ProductName":"Laptop","Value":"1500","Weight":"5.0"},
    {"ProductName":"Headphones","Value":"100","Weight":"0.5"},
    {"ProductName":"Camera","Value":"800","Weight":"2.0"},
    {"ProductName":"Smartwatch","Value":"250","Weight":"0.3"},
    {"ProductName":"Tablet","Value":"600","Weight":"1.5"},
    {"ProductName":"Bluetooth Speaker","Value":"150","Weight":"1.0"},
    {"ProductName":"Keyboard","Value":"80","Weight":"0.8"},
    {"ProductName":"Mouse","Value":"50","Weight":"0.2"},
    {"ProductName":"Monitor","Value":"300","Weight":"3.0"},
    {"ProductName":"Printer","Value":"400","Weight":"4.0"},
    {"ProductName":"External Hard Drive","Value":"120","Weight":"0.5"},
    {"ProductName":"Router","Value":"60","Weight":"0.3"},
    {"ProductName":"Power Bank","Value":"40","Weight":"0.4"},
    {"ProductName":"Memory Card","Value":"30","Weight":"0.05"},
    {"ProductName":"USB Flash Drive","Value":"25","Weight":"0.02"},
    {"ProductName":"Smart Home Hub","Value":"100","Weight":"0.6"},
    {"ProductName":"Gaming Console","Value":"500","Weight":"4.0"},
    {"ProductName":"Fitness Tracker","Value":"90","Weight":"0.2"},
    {"ProductName":"E-Reader","Value":"180","Weight":"0.5"}
  ]
}