Let the set of areas be indexed by i, with the following data from products.csv:

| i  | ProductName         | Value (benefit coefficient) | Weight (per-unit capacity consumption) |
|----|---------------------|----------------------------|----------------------------------------|
| 1  | Queens              | 443                        | 104                                    |
| 2  | Brooklyn            | 522                        | 368                                    |
| 3  | Manhattan           | 300                        | 483                                    |
| 4  | Bronx               | 767                        | 165                                    |
| 5  | Staten Island       | 300                        | 105                                    |
| 6  | Harlem              | 309                        | 123                                    |
| 7  | Upper East Side     | 598                        | 131                                    |
| 8  | Lower Manhattan     | 460                        | 341                                    |
| 9  | Midtown             | 318                        | 258                                    |
| 10 | Long Island City    | 126                        | 469                                    |
| 11 | Williamsburg        | 593                        | 387                                    |
| 12 | Bushwick            | 871                        | 425                                    |
| 13 | Flatbush            | 858                        | 482                                    |
| 14 | Greenpoint          | 321                        | 495                                    |
| 15 | Park Slope          | 275                        | 305                                    |
| 16 | Astoria             | 700                        | 377                                    |
| 17 | Jackson Heights     | 685                        | 318                                    |
| 18 | Flushing            | 940                        | 56                                     |
| 19 | Sunnyside           | 522                        | 213                                    |
| 20 | Ditmars             | 763                        | 472                                    |

Let x_i = daily scale of development in area i (integer, x_i ≥ 0).

The optimization model is:

Maximize:
  443·x_Queens + 522·x_Brooklyn + 300·x_Manhattan + 767·x_Bronx + 300·x_StatenIsland + 309·x_Harlem + 598·x_UpperEastSide + 460·x_LowerManhattan + 318·x_Midtown + 126·x_LongIslandCity + 593·x_Williamsburg + 871·x_Bushwick + 858·x_Flatbush + 321·x_Greenpoint + 275·x_ParkSlope + 700·x_Astoria + 685·x_JacksonHeights + 940·x_Flushing + 522·x_Sunnyside + 763·x_Ditmars

Subject to:
  104·x_Queens + 368·x_Brooklyn + 483·x_Manhattan + 165·x_Bronx + 105·x_StatenIsland + 123·x_Harlem + 131·x_UpperEastSide + 341·x_LowerManhattan + 258·x_Midtown + 469·x_LongIslandCity + 387·x_Williamsburg + 425·x_Bushwick + 482·x_Flatbush + 495·x_Greenpoint + 305·x_ParkSlope + 377·x_Astoria + 318·x_JacksonHeights + 56·x_Flushing + 213·x_Sunnyside + 472·x_Ditmars ≤ 4466

  x_i ∈ {0, 1, 2, ...} for all areas i listed above.

Where:
- The coefficients and area names are taken directly from the supplied products.csv, in the original row order.
- The total capacity (right-hand side) is 4466, as given in capacity.csv.
- All variables are nonnegative integers, as required by the user description.