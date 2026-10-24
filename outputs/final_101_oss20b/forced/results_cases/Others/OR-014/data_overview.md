{
  "query": "A restaurant is planning to sell a variety of pizzas, with the revenue details for each type provided in the ‘Revenue’ column of the dataset. Each pizza type has its own demand level. The objective is to maximize the total revenue based on the available initial inventory of all kinds of pizzas, as detailed in the ‘Initial Inventory’ column. During the sales horizon, demand for each pizza type is assumed to be deterministic and known in advance, with demand information specified in the ‘Demand’ column. The decision variables, denoted as x_i, represent the number of units of each pizza type to be fulfilled, where each decision variable is a non-negative integer.\n\nBecause the fulfillment quantity for each pizza type cannot exceed either the available initial inventory or the corresponding demand, the decision variables must satisfy both inventory and demand constraints. The resulting problem can therefore be formulated as a deterministic integer optimization model that determines the optimal fulfillment quantities in order to maximize total revenue.",
  "relationships": [],
  "tables": [
    {
      "columns": [
        "Product Name",
        "Revenue",
        "Demand",
        "Initial Inventory"
      ],
      "file_name": "PizzaSalesDataset.csv",
      "records": [
        {
          "data_row_number": 0,
          "values": {
            "Demand": "1960",
            "Initial Inventory": "9920",
            "Product Name": "bbq_ckn_l",
            "Revenue": "20.75"
          }
        },
        {
          "data_row_number": 1,
          "values": {
            "Demand": "1884",
            "Initial Inventory": "9560",
            "Product Name": "bbq_ckn_m",
            "Revenue": "16.75"
          }
        },
        {
          "data_row_number": 2,
          "values": {
            "Demand": "963",
            "Initial Inventory": "4840",
            "Product Name": "bbq_ckn_s",
            "Revenue": "12.75"
          }
        },
        {
          "data_row_number": 3,
          "values": {
            "Demand": "3729",
            "Initial Inventory": "19140",
            "Product Name": "big_meat_s",
            "Revenue": "12"
          }
        },
        {
          "data_row_number": 4,
          "values": {
            "Demand": "970",
            "Initial Inventory": "4900",
            "Product Name": "brie_carre_s",
            "Revenue": "23.65"
          }
        },
        {
          "data_row_number": 5,
          "values": {
            "Demand": "550",
            "Initial Inventory": "2760",
            "Product Name": "calabrese_l",
            "Revenue": "20.25"
          }
        },
        {
          "data_row_number": 6,
          "values": {
            "Demand": "1116",
            "Initial Inventory": "5620",
            "Product Name": "calabrese_m",
            "Revenue": "16.25"
          }
        },
        {
          "data_row_number": 7,
          "values": {
            "Demand": "198",
            "Initial Inventory": "990",
            "Product Name": "calabrese_s",
            "Revenue": "12.25"
          }
        },
        {
          "data_row_number": 8,
          "values": {
            "Demand": "1823",
            "Initial Inventory": "9270",
            "Product Name": "cali_ckn_l",
            "Revenue": "20.75"
          }
        },
        {
          "data_row_number": 9,
          "values": {
            "Demand": "1858",
            "Initial Inventory": "9440",
            "Product Name": "cali_ckn_m",
            "Revenue": "16.75"
          }
        },
        {
          "data_row_number": 10,
          "values": {
            "Demand": "992",
            "Initial Inventory": "4990",
            "Product Name": "cali_ckn_s",
            "Revenue": "12.75"
          }
        },
        {
          "data_row_number": 11,
          "values": {
            "Demand": "375",
            "Initial Inventory": "1880",
            "Product Name": "ckn_alfredo_l",
            "Revenue": "20.75"
          }
        },
        {
          "data_row_number": 12,
          "values": {
            "Demand": "1400",
            "Initial Inventory": "7030",
            "Product Name": "ckn_alfredo_m",
            "Revenue": "16.75"
          }
        },
        {
          "data_row_number": 13,
          "values": {
            "Demand": "192",
            "Initial Inventory": "960",
            "Product Name": "ckn_alfredo_s",
            "Revenue": "12.75"
          }
        },
        {
          "data_row_number": 14,
          "values": {
            "Demand": "791",
            "Initial Inventory": "3990",
            "Product Name": "ckn_pesto_l",
            "Revenue": "20.75"
          }
        },
        {
          "data_row_number": 15,
          "values": {
            "Demand": "550",
            "Initial Inventory": "2760",
            "Product Name": "ckn_pesto_m",
            "Revenue": "16.75"
          }
        },
        {
          "data_row_number": 16,
          "values": {
            "Demand": "593",
            "Initial Inventory": "2980",
            "Product Name": "ckn_pesto_s",
            "Revenue": "12.75"
          }
        },
        {
          "data_row_number": 17,
          "values": {
            "Demand": "944",
            "Initial Inventory": "4730",
            "Product Name": "classic_dlx_l",
            "Revenue": "20.5"
          }
        },
        {
          "data_row_number": 18,
          "values": {
            "Demand": "2340",
            "Initial Inventory": "11810",
            "Product Name": "classic_dlx_m",
            "Revenue": "16"
          }
        },
        {
          "data_row_number": 19,
          "values": {
            "Demand": "1586",
            "Initial Inventory": "7990",
            "Product Name": "classic_dlx_s",
            "Revenue": "12"
          }
        },
        {
          "data_row_number": 20,
          "values": {
            "Demand": "2769",
            "Initial Inventory": "14090",
            "Product Name": "five_cheese_l",
            "Revenue": "18.5"
          }
        },
        {
          "data_row_number": 21,
          "values": {
            "Demand": "2589",
            "Initial Inventory": "13160",
            "Product Name": "four_cheese_l",
            "Revenue": "17.95"
          }
        },
        {
          "data_row_number": 22,
          "values": {
            "Demand": "1163",
            "Initial Inventory": "5860",
            "Product Name": "four_cheese_m",
            "Revenue": "14.75"
          }
        },
        {
          "data_row_number": 23,
          "values": {
            "Demand": "189",
            "Initial Inventory": "950",
            "Product Name": "green_garden_l",
            "Revenue": "20.25"
          }
        },
        {
          "data_row_number": 24,
          "values": {
            "Demand": "602",
            "Initial Inventory": "3020",
            "Product Name": "green_garden_m",
            "Revenue": "16"
          }
        },
        {
          "data_row_number": 25,
          "values": {
            "Demand": "1193",
            "Initial Inventory": "6000",
            "Product Name": "green_garden_s",
            "Revenue": "12"
          }
        },
        {
          "data_row_number": 26,
          "values": {
            "Demand": "1815",
            "Initial Inventory": "9190",
            "Product Name": "hawaiian_l",
            "Revenue": "16.5"
          }
        },
        {
          "data_row_number": 27,
          "values": {
            "Demand": "956",
            "Initial Inventory": "4830",
            "Product Name": "hawaiian_m",
            "Revenue": "13.25"
          }
        },
        {
          "data_row_number": 28,
          "values": {
            "Demand": "2021",
            "Initial Inventory": "10200",
            "Product Name": "hawaiian_s",
            "Revenue": "10.5"
          }
        },
        {
          "data_row_number": 29,
          "values": {
            "Demand": "1447",
            "Initial Inventory": "7320",
            "Product Name": "ital_cpcllo_l",
            "Revenue": "20.5"
          }
        },
        {
          "data_row_number": 30,
          "values": {
            "Demand": "803",
            "Initial Inventory": "4040",
            "Product Name": "ital_cpcllo_m",
            "Revenue": "16"
          }
        },
        {
          "data_row_number": 31,
          "values": {
            "Demand": "602",
            "Initial Inventory": "3020",
            "Product Name": "ital_cpcllo_s",
            "Revenue": "12"
          }
        },
        {
          "data_row_number": 32,
          "values": {
            "Demand": "1482",
            "Initial Inventory": "7470",
            "Product Name": "ital_supr_l",
            "Revenue": "20.75"
          }
        },
        {
          "data_row_number": 33,
          "values": {
            "Demand": "1861",
            "Initial Inventory": "9410",
            "Product Name": "ital_supr_m",
            "Revenue": "16.5"
          }
        },
        {
          "data_row_number": 34,
          "values": {
            "Demand": "390",
            "Initial Inventory": "1960",
            "Product Name": "ital_supr_s",
            "Revenue": "12.5"
          }
        },
        {
          "data_row_number": 35,
          "values": {
            "Demand": "380",
            "Initial Inventory": "1900",
            "Product Name": "ital_veggie_l",
            "Revenue": "21"
          }
        },
        {
          "data_row_number": 36,
          "values": {
            "Demand": "969",
            "Initial Inventory": "4860",
            "Product Name": "ital_veggie_m",
            "Revenue": "16.75"
          }
        },
        {
          "data_row_number": 37,
          "values": {
            "Demand": "607",
            "Initial Inventory": "3050",
            "Product Name": "ital_veggie_s",
            "Revenue": "12.75"
          }
        },
        {
          "data_row_number": 38,
          "values": {
            "Demand": "734",
            "Initial Inventory": "3700",
            "Product Name": "mediterraneo_l",
            "Revenue": "20.25"
          }
        },
        {
          "data_row_number": 39,
          "values": {
            "Demand": "546",
            "Initial Inventory": "2750",
            "Product Name": "mediterraneo_m",
            "Revenue": "16"
          }
        },
        {
          "data_row_number": 40,
          "values": {
            "Demand": "577",
            "Initial Inventory": "2890",
            "Product Name": "mediterraneo_s",
            "Revenue": "12"
          }
        },
        {
          "data_row_number": 41,
          "values": {
            "Demand": "1711",
            "Initial Inventory": "8670",
            "Product Name": "mexicana_l",
            "Revenue": "20.25"
          }
        },
        {
          "data_row_number": 42,
          "values": {
            "Demand": "907",
            "Initial Inventory": "4550",
            "Product Name": "mexicana_m",
            "Revenue": "16"
          }
        },
        {
          "data_row_number": 43,
          "values": {
            "Demand": "322",
            "Initial Inventory": "1620",
            "Product Name": "mexicana_s",
            "Revenue": "12"
          }
        },
        {
          "data_row_number": 44,
          "values": {
            "Demand": "1123",
            "Initial Inventory": "5660",
            "Product Name": "napolitana_l",
            "Revenue": "20.5"
          }
        },
        {
          "data_row_number": 45,
          "values": {
            "Demand": "853",
            "Initial Inventory": "4270",
            "Product Name": "napolitana_m",
            "Revenue": "16"
          }
        },
        {
          "data_row_number": 46,
          "values": {
            "Demand": "939",
            "Initial Inventory": "4710",
            "Product Name": "napolitana_s",
            "Revenue": "12"
          }
        },
        {
          "data_row_number": 47,
          "values": {
            "Demand": "765",
            "Initial Inventory": "3840",
            "Product Name": "pep_msh_pep_l",
            "Revenue": "17.5"
          }
        },
        {
          "data_row_number": 48,
          "values": {
            "Demand": "788",
            "Initial Inventory": "3970",
            "Product Name": "pep_msh_pep_m",
            "Revenue": "14.5"
          }
        },
        {
          "data_row_number": 49,
          "values": {
            "Demand": "1148",
            "Initial Inventory": "5780",
            "Product Name": "pep_msh_pep_s",
            "Revenue": "11"
          }
        },
        {
          "data_row_number": 50,
          "values": {
            "Demand": "1440",
            "Initial Inventory": "7280",
            "Product Name": "pepperoni_l",
            "Revenue": "15.25"
          }
        },
        {
          "data_row_number": 51,
          "values": {
            "Demand": "1857",
            "Initial Inventory": "9390",
            "Product Name": "pepperoni_m",
            "Revenue": "12.5"
          }
        },
        {
          "data_row_number": 52,
          "values": {
            "Demand": "1490",
            "Initial Inventory": "7510",
            "Product Name": "pepperoni_s",
            "Revenue": "9.75"
          }
        },
        {
          "data_row_number": 53,
          "values": {
            "Demand": "1376",
            "Initial Inventory": "6960",
            "Product Name": "peppr_salami_l",
            "Revenue": "20.75"
          }
        },
        {
          "data_row_number": 54,
          "values": {
            "Demand": "852",
            "Initial Inventory": "4280",
            "Product Name": "peppr_salami_m",
            "Revenue": "16.5"
          }
        },
        {
          "data_row_number": 55,
          "values": {
            "Demand": "640",
            "Initial Inventory": "3220",
            "Product Name": "peppr_salami_s",
            "Revenue": "12.5"
          }
        },
        {
          "data_row_number": 56,
          "values": {
            "Demand": "858",
            "Initial Inventory": "4350",
            "Product Name": "prsc_argla_l",
            "Revenue": "20.75"
          }
        },
        {
          "data_row_number": 57,
          "values": {
            "Demand": "1183",
            "Initial Inventory": "5980",
            "Product Name": "prsc_argla_m",
            "Revenue": "16.5"
          }
        },
        {
          "data_row_number": 58,
          "values": {
            "Demand": "844",
            "Initial Inventory": "4240",
            "Product Name": "prsc_argla_s",
            "Revenue": "12.5"
          }
        },
        {
          "data_row_number": 59,
          "values": {
            "Demand": "1209",
            "Initial Inventory": "6130",
            "Product Name": "sicilian_l",
            "Revenue": "20.25"
          }
        },
        {
          "data_row_number": 60,
          "values": {
            "Demand": "1135",
            "Initial Inventory": "5740",
            "Product Name": "sicilian_m",
            "Revenue": "16.25"
          }
        },
        {
          "data_row_number": 61,
          "values": {
            "Demand": "1482",
            "Initial Inventory": "7510",
            "Product Name": "sicilian_s",
            "Revenue": "12.25"
          }
        },
        {
          "data_row_number": 62,
          "values": {
            "Demand": "806",
            "Initial Inventory": "4050",
            "Product Name": "soppressata_l",
            "Revenue": "20.75"
          }
        },
        {
          "data_row_number": 63,
          "values": {
            "Demand": "536",
            "Initial Inventory": "2680",
            "Product Name": "soppressata_m",
            "Revenue": "16.5"
          }
        },
        {
          "data_row_number": 64,
          "values": {
            "Demand": "576",
            "Initial Inventory": "2880",
            "Product Name": "soppressata_s",
            "Revenue": "12.5"
          }
        },
        {
          "data_row_number": 65,
          "values": {
            "Demand": "2009",
            "Initial Inventory": "10160",
            "Product Name": "southw_ckn_l",
            "Revenue": "20.75"
          }
        },
        {
          "data_row_number": 66,
          "values": {
            "Demand": "1060",
            "Initial Inventory": "5340",
            "Product Name": "southw_ckn_m",
            "Revenue": "16.75"
          }
        },
        {
          "data_row_number": 67,
          "values": {
            "Demand": "733",
            "Initial Inventory": "3670",
            "Product Name": "southw_ckn_s",
            "Revenue": "12.75"
          }
        },
        {
          "data_row_number": 68,
          "values": {
            "Demand": "2198",
            "Initial Inventory": "11090",
            "Product Name": "spicy_ital_l",
            "Revenue": "20.75"
          }
        },
        {
          "data_row_number": 69,
          "values": {
            "Demand": "808",
            "Initial Inventory": "4080",
            "Product Name": "spicy_ital_m",
            "Revenue": "16.5"
          }
        },
        {
          "data_row_number": 70,
          "values": {
            "Demand": "807",
            "Initial Inventory": "4070",
            "Product Name": "spicy_ital_s",
            "Revenue": "12.5"
          }
        },
        {
          "data_row_number": 71,
          "values": {
            "Demand": "563",
            "Initial Inventory": "2840",
            "Product Name": "spin_pesto_l",
            "Revenue": "20.75"
          }
        },
        {
          "data_row_number": 72,
          "values": {
            "Demand": "563",
            "Initial Inventory": "2820",
            "Product Name": "spin_pesto_m",
            "Revenue": "16.5"
          }
        },
        {
          "data_row_number": 73,
          "values": {
            "Demand": "801",
            "Initial Inventory": "4040",
            "Product Name": "spin_pesto_s",
            "Revenue": "12.5"
          }
        },
        {
          "data_row_number": 74,
          "values": {
            "Demand": "882",
            "Initial Inventory": "4450",
            "Product Name": "spinach_fet_l",
            "Revenue": "20.25"
          }
        },
        {
          "data_row_number": 75,
          "values": {
            "Demand": "1120",
            "Initial Inventory": "5620",
            "Product Name": "spinach_fet_m",
            "Revenue": "16"
          }
        },
        {
          "data_row_number": 76,
          "values": {
            "Demand": "876",
            "Initial Inventory": "4390",
            "Product Name": "spinach_fet_s",
            "Revenue": "12"
          }
        },
        {
          "data_row_number": 77,
          "values": {
            "Demand": "563",
            "Initial Inventory": "2830",
            "Product Name": "spinach_supr_l",
            "Revenue": "20.75"
          }
        },
        {
          "data_row_number": 78,
          "values": {
            "Demand": "533",
            "Initial Inventory": "2670",
            "Product Name": "spinach_supr_m",
            "Revenue": "16.5"
          }
        },
        {
          "data_row_number": 79,
          "values": {
            "Demand": "794",
            "Initial Inventory": "4000",
            "Product Name": "spinach_supr_s",
            "Revenue": "12.5"
          }
        },
        {
          "data_row_number": 80,
          "values": {
            "Demand": "2776",
            "Initial Inventory": "14100",
            "Product Name": "thai_ckn_l",
            "Revenue": "20.75"
          }
        },
        {
          "data_row_number": 81,
          "values": {
            "Demand": "955",
            "Initial Inventory": "4810",
            "Product Name": "thai_ckn_m",
            "Revenue": "16.75"
          }
        },
        {
          "data_row_number": 82,
          "values": {
            "Demand": "956",
            "Initial Inventory": "4800",
            "Product Name": "thai_ckn_s",
            "Revenue": "12.75"
          }
        },
        {
          "data_row_number": 83,
          "values": {
            "Demand": "510",
            "Initial Inventory": "2550",
            "Product Name": "the_greek_l",
            "Revenue": "20.5"
          }
        },
        {
          "data_row_number": 84,
          "values": {
            "Demand": "560",
            "Initial Inventory": "2810",
            "Product Name": "the_greek_m",
            "Revenue": "16"
          }
        },
        {
          "data_row_number": 85,
          "values": {
            "Demand": "604",
            "Initial Inventory": "3040",
            "Product Name": "the_greek_s",
            "Revenue": "12"
          }
        },
        {
          "data_row_number": 86,
          "values": {
            "Demand": "1096",
            "Initial Inventory": "5520",
            "Product Name": "the_greek_xl",
            "Revenue": "25.5"
          }
        },
        {
          "data_row_number": 87,
          "values": {
            "Demand": "56",
            "Initial Inventory": "280",
            "Product Name": "the_greek_xxl",
            "Revenue": "35.95"
          }
        },
        {
          "data_row_number": 88,
          "values": {
            "Demand": "850",
            "Initial Inventory": "4270",
            "Product Name": "veggie_veg_l",
            "Revenue": "20.25"
          }
        },
        {
          "data_row_number": 89,
          "values": {
            "Demand": "1265",
            "Initial Inventory": "6350",
            "Product Name": "veggie_veg_m",
            "Revenue": "16"
          }
        },
        {
          "data_row_number": 90,
          "values": {
            "Demand": "921",
            "Initial Inventory": "4640",
            "Product Name": "veggie_veg_s",
            "Revenue": "12"
          }
        },
        {
          "data_row_number": 91,
          "values": {
            "Demand": "733",
            "Initial Inventory": "2830",
            "Product Name": "fried_duck_l",
            "Revenue": "16.5"
          }
        },
        {
          "data_row_number": 92,
          "values": {
            "Demand": "2198",
            "Initial Inventory": "2670",
            "Product Name": "fried_duck_m",
            "Revenue": "12.5"
          }
        },
        {
          "data_row_number": 93,
          "values": {
            "Demand": "808",
            "Initial Inventory": "4000",
            "Product Name": "fried_duck_s",
            "Revenue": "21"
          }
        },
        {
          "data_row_number": 94,
          "values": {
            "Demand": "563",
            "Initial Inventory": "2680",
            "Product Name": "mapo_Tofu_l",
            "Revenue": "16.75"
          }
        },
        {
          "data_row_number": 95,
          "values": {
            "Demand": "801",
            "Initial Inventory": "2880",
            "Product Name": "mapo_Tofu_m",
            "Revenue": "12.75"
          }
        },
        {
          "data_row_number": 96,
          "values": {
            "Demand": "882",
            "Initial Inventory": "10160",
            "Product Name": "mapo_Tofu_s",
            "Revenue": "20.25"
          }
        },
        {
          "data_row_number": 97,
          "values": {
            "Demand": "1120",
            "Initial Inventory": "2820",
            "Product Name": "sea_food_l",
            "Revenue": "16"
          }
        },
        {
          "data_row_number": 98,
          "values": {
            "Demand": "794",
            "Initial Inventory": "4040",
            "Product Name": "sea_food_m",
            "Revenue": "20.75"
          }
        },
        {
          "data_row_number": 99,
          "values": {
            "Demand": "2776",
            "Initial Inventory": "4450",
            "Product Name": "sea_food_s",
            "Revenue": "16.5"
          }
        }
      ],
      "table_id": "file_0_view_0"
    }
  ]
}
