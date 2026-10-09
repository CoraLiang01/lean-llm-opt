##### Decision Variables

Let $x_i$ denote the number of units of product $i$ to be fulfilled, for each product $i$ in the set of all products.

##### Objective Function

$\max \sum_{i} r_i x_i$

where $r_i$ is the unit revenue for product $i$.

##### Constraints

1. **Inventory and Demand Constraints:**

For each product $i$:

$0 \leq x_i \leq \min\{\text{Initial Inventory}_i, \text{Demand}_i\}$

2. **Variable Domain:**

$x_i$ is an integer, for all $i$.

##### Retrieved Information

{
  "products": [
    {
      "Full_Product_Name": "Butter_Amul",
      "Revenue": 96.86,
      "Demand": 34102,
      "Initial Inventory": 29862
    },
    {
      "Full_Product_Name": "Butter_Mother Dairy",
      "Revenue": 48.01,
      "Demand": 36579,
      "Initial Inventory": 29898
    },
    {
      "Full_Product_Name": "Butter_Parag Milk Foods",
      "Revenue": 8.83,
      "Demand": 36086,
      "Initial Inventory": 25208
    },
    {
      "Full_Product_Name": "Butter_Warana",
      "Revenue": 92.96,
      "Demand": 41254,
      "Initial Inventory": 30816
    },
    {
      "Full_Product_Name": "Buttermilk_Amul",
      "Revenue": 40.75,
      "Demand": 29876,
      "Initial Inventory": 19925
    },
    {
      "Full_Product_Name": "Buttermilk_Mother Dairy",
      "Revenue": 83.07,
      "Demand": 41229,
      "Initial Inventory": 26482
    },
    {
      "Full_Product_Name": "Buttermilk_Raj",
      "Revenue": 15.64,
      "Demand": 35354,
      "Initial Inventory": 30865
    },
    {
      "Full_Product_Name": "Buttermilk_Sudha",
      "Revenue": 56.57,
      "Demand": 29649,
      "Initial Inventory": 33517
    },
    {
      "Full_Product_Name": "Cheese_Amul",
      "Revenue": 100.74,
      "Demand": 38558,
      "Initial Inventory": 30929
    },
    {
      "Full_Product_Name": "Cheese_Britannia Industries",
      "Revenue": 28.92,
      "Demand": 28603,
      "Initial Inventory": 21405
    },
    {
      "Full_Product_Name": "Cheese_Dynamix Dairies",
      "Revenue": 32.66,
      "Demand": 35962,
      "Initial Inventory": 25953
    },
    {
      "Full_Product_Name": "Cheese_Passion Cheese",
      "Revenue": 58.09,
      "Demand": 36961,
      "Initial Inventory": 23825
    },
    {
      "Full_Product_Name": "Curd_Amul",
      "Revenue": 30.27,
      "Demand": 39436,
      "Initial Inventory": 31687
    },
    {
      "Full_Product_Name": "Curd_Mother Dairy",
      "Revenue": 84.57,
      "Demand": 43522,
      "Initial Inventory": 33377
    },
    {
      "Full_Product_Name": "Curd_Raj",
      "Revenue": 84.75,
      "Demand": 38128,
      "Initial Inventory": 34914
    },
    {
      "Full_Product_Name": "Curd_Sudha",
      "Revenue": 76.37,
      "Demand": 42341,
      "Initial Inventory": 33547
    },
    {
      "Full_Product_Name": "Ghee_Amul",
      "Revenue": 41.49,
      "Demand": 30345,
      "Initial Inventory": 23120
    },
    {
      "Full_Product_Name": "Ghee_Mother Dairy",
      "Revenue": 52.79,
      "Demand": 35420,
      "Initial Inventory": 24667
    },
    {
      "Full_Product_Name": "Ghee_Raj",
      "Revenue": 48.13,
      "Demand": 34100,
      "Initial Inventory": 25395
    },
    {
      "Full_Product_Name": "Ghee_Sudha",
      "Revenue": 95.09,
      "Demand": 33007,
      "Initial Inventory": 24676
    },
    {
      "Full_Product_Name": "Ice Cream_Amul",
      "Revenue": 54.41,
      "Demand": 37894,
      "Initial Inventory": 26707
    },
    {
      "Full_Product_Name": "Ice Cream_Dodla Dairy",
      "Revenue": 82.24,
      "Demand": 29840,
      "Initial Inventory": 26722
    },
    {
      "Full_Product_Name": "Ice Cream_Mother Dairy",
      "Revenue": 94.32,
      "Demand": 38762,
      "Initial Inventory": 25809
    },
    {
      "Full_Product_Name": "Ice Cream_Palle2patnam",
      "Revenue": 83.73,
      "Demand": 34674,
      "Initial Inventory": 24391
    },
    {
      "Full_Product_Name": "Lassi_Amul",
      "Revenue": 74.45,
      "Demand": 42972,
      "Initial Inventory": 30728
    },
    {
      "Full_Product_Name": "Lassi_Mother Dairy",
      "Revenue": 49.4,
      "Demand": 33894,
      "Initial Inventory": 28628
    },
    {
      "Full_Product_Name": "Lassi_Raj",
      "Revenue": 93.93,
      "Demand": 45762,
      "Initial Inventory": 30568
    },
    {
      "Full_Product_Name": "Lassi_Sudha",
      "Revenue": 88.05,
      "Demand": 29503,
      "Initial Inventory": 23461
    },
    {
      "Full_Product_Name": "Milk_Amul",
      "Revenue": 39.24,
      "Demand": 34761,
      "Initial Inventory": 21398
    },
    {
      "Full_Product_Name": "Milk_Mother Dairy",
      "Revenue": 8.69,
      "Demand": 40548,
      "Initial Inventory": 33619
    },
    {
      "Full_Product_Name": "Milk_Raj",
      "Revenue": 65.53,
      "Demand": 43012,
      "Initial Inventory": 26355
    },
    {
      "Full_Product_Name": "Milk_Sudha",
      "Revenue": 42.34,
      "Demand": 29180,
      "Initial Inventory": 23815
    },
    {
      "Full_Product_Name": "Paneer_Amul",
      "Revenue": 81.76,
      "Demand": 33498,
      "Initial Inventory": 20787
    },
    {
      "Full_Product_Name": "Paneer_Mother Dairy",
      "Revenue": 29.09,
      "Demand": 34848,
      "Initial Inventory": 29342
    },
    {
      "Full_Product_Name": "Paneer_Raj",
      "Revenue": 87.3,
      "Demand": 40347,
      "Initial Inventory": 23556
    },
    {
      "Full_Product_Name": "Paneer_Sudha",
      "Revenue": 66.7,
      "Demand": 37188,
      "Initial Inventory": 28753
    },
    {
      "Full_Product_Name": "Yogurt_Amul",
      "Revenue": 89.32,
      "Demand": 34347,
      "Initial Inventory": 24404
    },
    {
      "Full_Product_Name": "Yogurt_Dodla Dairy",
      "Revenue": 33.81,
      "Demand": 37181,
      "Initial Inventory": 26829
    },
    {
      "Full_Product_Name": "Yogurt_Mother Dairy",
      "Revenue": 25.29,
      "Demand": 36644,
      "Initial Inventory": 25562
    },
    {
      "Full_Product_Name": "Yogurt_Palle2patnam",
      "Revenue": 84.9,
      "Demand": 34303,
      "Initial Inventory": 28695
    }
  ]
}

##### Model Summary

For each product $i$ (see list above for all 40 products), let $x_i$ be the number of units fulfilled.

$\max \sum_{i} r_i x_i$

subject to

$0 \leq x_i \leq \min\{\text{Initial Inventory}_i, \text{Demand}_i\}$, $x_i$ integer, for all $i$.