**Mathematical Model**

```
SETS
    P   :  set of products

PARAMETERS
    Labor_p      {p ∈ P}   : labor required per unit of product p
    Material_p   {p ∈ P}   : material required per unit of product p
    Price_p      {p ∈ P}   : selling price per unit of product p
    VarCost_p    {p ∈ P}   : variable cost per unit of product p

CONSTANTS
    LaborCap     = 1650          : weekly labor capacity
    MaterialCap  = 1850          : weekly material capacity
    FixedCost    = 4500          : fixed operating cost per week

VARIABLES
    x_p          {p ∈ P}   : production quantity of product p   (continuous, ≥ 0)

MINIMIZE / MAXIMIZE
    Z = Σ_{p∈P} (Price_p - VarCost_p) * x_p  -  FixedCost

CONSTRAINTS
    LaborCon   : Σ_{p∈P} Labor_p * x_p   ≤  LaborCap
    MaterialCon: Σ_{p∈P} Material_p * x_p ≤  MaterialCap
    NonNeg     : x_p ≥ 0   ∀ p ∈ P
```

**Data Mapping (CSVQA_DATA)**  

| Symbol | Source Table | Column Name |
|--------|--------------|-------------|
| P      | file_0_view_0 | “Product Name” |
| Labor_p | file_0_view_0 | “Labor per unit” |
| Material_p | file_0_view_0 | “Material per unit” |
| Price_p | file_0_view_0 | “Selling Price” |
| VarCost_p | file_0_view_0 | “Variable Cost” |

*All other constants (LaborCap, MaterialCap, FixedCost) are fixed numeric values supplied in the problem statement.*