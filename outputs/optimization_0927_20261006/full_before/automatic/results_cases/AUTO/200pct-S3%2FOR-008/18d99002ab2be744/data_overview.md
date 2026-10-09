Here is all data required to formulate the model from "products.csv" and "capacity.csv", preserving source order and all identifiers and coefficients:

From "capacity.csv":

| VehicleID | VehicleType         | Capacity | previous_period_inventory_status | previous_period_capacity | capacity_two_periods_ago | inventory_status_two_periods_ago | inventory_status_previous_year | capacity_previous_year |
|-----------|--------------------|----------|----------------------------------|-------------------------|--------------------------|-----------------------------------|-------------------------------|-----------------------|
| 1         | Sedans             | 100      | Stockout                         | 110                     | 116                      | Overstock                         | Stockout                      | 118                   |
| 2         | SUVs               | 80       | Overstock                        | 73                      | 66                       | Stockout                          | Overstock                     | 78                    |
| 3         | Electric Vehicles  | 120      | Overstock                        | 141                     | 123                      | Overstock                         | Balanced                      | 121                   |
| 4         | Hybrid Vehicles    | 90       | Overstock                        | 87                      | 80                       | Overstock                         | Overstock                     | 96                    |
| 5         | Trucks             | 50       | Stockout                         | 60                      | 41                       | Stockout                          | Balanced                      | 43                    |
| 6         | Sports Cars        | 30       | Balanced                         | 27                      | 32                       | Overstock                         | Balanced                      | 24                    |
| 7         | Compact Cars       | 110      | Overstock                        | 116                     | 109                      | Stockout                          | Balanced                      | 115                   |
| 8         | Luxury Sedans      | 40       | Overstock                        | 33                      | 43                       | Balanced                          | Overstock                     | 32                    |
| 9         | Vans               | 60       | Balanced                         | 61                      | 69                       | Stockout                          | Stockout                      | 71                    |
| 10        | Pickup Trucks      | 35       | Stockout                         | 32                      | 36                       | Balanced                          | Overstock                     | 36                    |

From "products.csv":

| ProductName         | Value | previous_period_unit_value | two_periods_ago_unit_value | previous_year_unit_value | next_year_unit_value_forecast |
|---------------------|-------|---------------------------|---------------------------|-------------------------|------------------------------|
| Sedans              | 1200  | 1366                      | 1388                      | 1014                    | 1117                         |
| SUVs                | 1800  | 2006                      | 2070                      | 1537                    | 1681                         |
| Electric Vehicles   | 2500  | 2088                      | 2528                      | 2904                    | 2335                         |
| Hybrid Vehicles     | 2000  | 1875                      | 1748                      | 2282                    | 1615                         |
| Trucks              | 1500  | 1640                      | 1744                      | 1665                    | 1430                         |
| Sports Cars         | 3000  | 3292                      | 3584                      | 3486                    | 2905                         |
| Compact Cars        | 1000  | 854                       | 1090                      | 852                     | 1103                         |
| Luxury Sedans       | 3500  | 3375                      | 3809                      | 3949                    | 4188                         |
| Vans                | 1600  | 1783                      | 1481                      | 1852                    | 1856                         |
| Pickup Trucks       | 1700  | 1874                      | 1432                      | 1574                    | 1862                         |