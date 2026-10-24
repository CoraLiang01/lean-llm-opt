{
  "query": "Amazon operates multiple distribution centers that are responsible for supplying essential goods daily to various customer groups. The daily demand for each customer group is provided in \"customer_demand.csv\", while the daily supply capacity of the distribution centers is detailed in \"supply_capacity.csv\". The transportation cost per unit of goods from each distribution center to each customer group is recorded in \"transportation_costs.csv\". The objective is to determine the quantity of goods to be transported from each distribution center to each customer group, ensuring that all customer demands are met without exceeding the supply capacity of any distribution center, while minimizing the total transportation cost.",
  "relationships": [],
  "tables": [
    {
      "columns": [
        "customer",
        "demand"
      ],
      "file_name": "customer_demand.csv",
      "records": [
        {
          "data_row_number": 0,
          "values": {
            "customer": "C1",
            "demand": "4415"
          }
        },
        {
          "data_row_number": 1,
          "values": {
            "customer": "C2",
            "demand": "5430"
          }
        },
        {
          "data_row_number": 2,
          "values": {
            "customer": "C3",
            "demand": "81"
          }
        },
        {
          "data_row_number": 3,
          "values": {
            "customer": "C4",
            "demand": "146"
          }
        },
        {
          "data_row_number": 4,
          "values": {
            "customer": "C5",
            "demand": "10638"
          }
        },
        {
          "data_row_number": 5,
          "values": {
            "customer": "C6",
            "demand": "1663"
          }
        },
        {
          "data_row_number": 6,
          "values": {
            "customer": "C7",
            "demand": "151"
          }
        },
        {
          "data_row_number": 7,
          "values": {
            "customer": "C8",
            "demand": "185"
          }
        },
        {
          "data_row_number": 8,
          "values": {
            "customer": "C9",
            "demand": "1917"
          }
        },
        {
          "data_row_number": 9,
          "values": {
            "customer": "C10",
            "demand": "4489"
          }
        },
        {
          "data_row_number": 10,
          "values": {
            "customer": "C11",
            "demand": "76"
          }
        },
        {
          "data_row_number": 11,
          "values": {
            "customer": "C12",
            "demand": "2529"
          }
        },
        {
          "data_row_number": 12,
          "values": {
            "customer": "C13",
            "demand": "2136"
          }
        },
        {
          "data_row_number": 13,
          "values": {
            "customer": "C14",
            "demand": "909"
          }
        },
        {
          "data_row_number": 14,
          "values": {
            "customer": "C15",
            "demand": "316"
          }
        },
        {
          "data_row_number": 15,
          "values": {
            "customer": "C16",
            "demand": "70"
          }
        },
        {
          "data_row_number": 16,
          "values": {
            "customer": "C17",
            "demand": "1456"
          }
        },
        {
          "data_row_number": 17,
          "values": {
            "customer": "C18",
            "demand": "2711"
          }
        }
      ],
      "table_id": "file_0_view_0"
    },
    {
      "columns": [
        "Unnamed: 0",
        "supply_capacity"
      ],
      "file_name": "supply_capacity.csv",
      "records": [
        {
          "data_row_number": 0,
          "values": {
            "Unnamed: 0": "S1",
            "supply_capacity": "5963"
          }
        },
        {
          "data_row_number": 1,
          "values": {
            "Unnamed: 0": "S2",
            "supply_capacity": "702"
          }
        },
        {
          "data_row_number": 2,
          "values": {
            "Unnamed: 0": "S3",
            "supply_capacity": "350"
          }
        },
        {
          "data_row_number": 3,
          "values": {
            "Unnamed: 0": "S4",
            "supply_capacity": "11483"
          }
        },
        {
          "data_row_number": 4,
          "values": {
            "Unnamed: 0": "S5",
            "supply_capacity": "6585"
          }
        },
        {
          "data_row_number": 5,
          "values": {
            "Unnamed: 0": "S6",
            "supply_capacity": "11330"
          }
        },
        {
          "data_row_number": 6,
          "values": {
            "Unnamed: 0": "S7",
            "supply_capacity": "207"
          }
        },
        {
          "data_row_number": 7,
          "values": {
            "Unnamed: 0": "S8",
            "supply_capacity": "788"
          }
        },
        {
          "data_row_number": 8,
          "values": {
            "Unnamed: 0": "S9",
            "supply_capacity": "6967"
          }
        },
        {
          "data_row_number": 9,
          "values": {
            "Unnamed: 0": "S10",
            "supply_capacity": "43"
          }
        },
        {
          "data_row_number": 10,
          "values": {
            "Unnamed: 0": "S11",
            "supply_capacity": "1137"
          }
        },
        {
          "data_row_number": 11,
          "values": {
            "Unnamed: 0": "S12",
            "supply_capacity": "1553"
          }
        },
        {
          "data_row_number": 12,
          "values": {
            "Unnamed: 0": "S13",
            "supply_capacity": "257"
          }
        },
        {
          "data_row_number": 13,
          "values": {
            "Unnamed: 0": "S14",
            "supply_capacity": "2114"
          }
        },
        {
          "data_row_number": 14,
          "values": {
            "Unnamed: 0": "S15",
            "supply_capacity": "205"
          }
        },
        {
          "data_row_number": 15,
          "values": {
            "Unnamed: 0": "S16",
            "supply_capacity": "17326"
          }
        },
        {
          "data_row_number": 16,
          "values": {
            "Unnamed: 0": "S17",
            "supply_capacity": "22260"
          }
        },
        {
          "data_row_number": 17,
          "values": {
            "Unnamed: 0": "S18",
            "supply_capacity": "333"
          }
        }
      ],
      "table_id": "file_1_view_0"
    },
    {
      "columns": [
        "Unnamed: 0",
        "C1",
        "C2",
        "C3",
        "C4"
      ],
      "file_name": "transportation_costs.csv",
      "records": [
        {
          "data_row_number": 0,
          "values": {
            "C1": "159.83765495858208",
            "C2": "6.42633790337285",
            "C3": "7.582243029047409",
            "C4": "99.52254454692138",
            "Unnamed: 0": "S1"
          }
        },
        {
          "data_row_number": 1,
          "values": {
            "C1": "0.10655701710223901",
            "C2": "8.54894494648944",
            "C3": "0.056878930310209665",
            "C4": "212.07495813173495",
            "Unnamed: 0": "S2"
          }
        },
        {
          "data_row_number": 2,
          "values": {
            "C1": "0.07121132969220138",
            "C2": "180.3996497782258",
            "C3": "0.26229720469556855",
            "C4": "182.8636027839307",
            "Unnamed: 0": "S3"
          }
        },
        {
          "data_row_number": 3,
          "values": {
            "C1": "1.5455382897983385",
            "C2": "180.07836268028748",
            "C3": "0.024056935459247156",
            "C4": "182.81622784995758",
            "Unnamed: 0": "S4"
          }
        },
        {
          "data_row_number": 4,
          "values": {
            "C1": "16.41408384079902",
            "C2": "20.385700354184078",
            "C3": "16.413172756532582",
            "C4": "478.03900704896546",
            "Unnamed: 0": "S5"
          }
        },
        {
          "data_row_number": 5,
          "values": {
            "C1": "158.9313402279755",
            "C2": "27.846407834093128",
            "C3": "7.505708371284356",
            "C4": "237.3730089947848",
            "Unnamed: 0": "S6"
          }
        },
        {
          "data_row_number": 6,
          "values": {
            "C1": "0.10720291412027193",
            "C2": "154.1398544913415",
            "C3": "0.8852806020632731",
            "C4": "183.15061946024696",
            "Unnamed: 0": "S7"
          }
        },
        {
          "data_row_number": 7,
          "values": {
            "C1": "2.220199373580392",
            "C2": "227.15154482810524",
            "C3": "47.959261803283134",
            "C4": "220.05727166867416",
            "Unnamed: 0": "S8"
          }
        },
        {
          "data_row_number": 8,
          "values": {
            "C1": "0.059394016843197485",
            "C2": "180.7637642245803",
            "C3": "0.0245032552848036",
            "C4": "213.5491059653967",
            "Unnamed: 0": "S9"
          }
        },
        {
          "data_row_number": 9,
          "values": {
            "C1": "76.00470976255896",
            "C2": "254.7550116921857",
            "C3": "66.2904204966841",
            "C4": "223.5475387800311",
            "Unnamed: 0": "S10"
          }
        },
        {
          "data_row_number": 10,
          "values": {
            "C1": "1349.7946298063491",
            "C2": "1031.0578018743602",
            "C3": "1348.9789699758965",
            "C4": "1179.2355621422678",
            "Unnamed: 0": "S11"
          }
        },
        {
          "data_row_number": 11,
          "values": {
            "C1": "1.435340983528516",
            "C2": "8.636406888143583",
            "C3": "0.07382328011512666",
            "C4": "10.215881024967333",
            "Unnamed: 0": "S12"
          }
        },
        {
          "data_row_number": 12,
          "values": {
            "C1": "314.2033136744687",
            "C2": "181.3919171146923",
            "C3": "268.5340851497754",
            "C4": "10.014234424202899",
            "Unnamed: 0": "S13"
          }
        },
        {
          "data_row_number": 13,
          "values": {
            "C1": "5.752413335760744",
            "C2": "7.158798027896069",
            "C3": "5.732884238005308",
            "C4": "89.16252252077103",
            "Unnamed: 0": "S14"
          }
        },
        {
          "data_row_number": 14,
          "values": {
            "C1": "654.160873808039",
            "C2": "44.28920565984101",
            "C3": "764.1914776236828",
            "C4": "43.60146761460413",
            "Unnamed: 0": "S15"
          }
        },
        {
          "data_row_number": 15,
          "values": {
            "C1": "10.834014840644409",
            "C2": "3.0964091452585922",
            "C3": "226.18066701518617",
            "C4": "256.0839847043925",
            "Unnamed: 0": "S16"
          }
        },
        {
          "data_row_number": 16,
          "values": {
            "C1": "16.409641338621462",
            "C2": "20.385033708139762",
            "C3": "16.408761812170813",
            "C4": "26.553380437646684",
            "Unnamed: 0": "S17"
          }
        },
        {
          "data_row_number": 17,
          "values": {
            "C1": "837.3389315505575",
            "C2": "47.44491934576151",
            "C3": "39.91671898599663",
            "C4": "47.691576615292604",
            "Unnamed: 0": "S18"
          }
        }
      ],
      "table_id": "file_2_view_0"
    }
  ]
}
