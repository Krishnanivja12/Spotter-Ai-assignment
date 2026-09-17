# Phase 2 — Exploratory Data Analysis

All values in this report are calculated from the supplied CSV files.

## Dataset dimensions and dates

- train-test: 48,000 rows x 14 columns
- validation: 12,000 rows x 13 columns
- train date range: 2025-01-01 to 2025-10-31 (304 unique dates)
- validation date range: 2025-11-01 to 2025-12-31 (61 unique dates)
- train rows by month:
```
date
2025-01    4918
2025-02    4337
2025-03    5036
2025-04    4819
2025-05    4913
2025-06    4783
2025-07    4912
2025-08    4759
2025-09    4670
2025-10    4853
Freq: M
```
- validation rows by month:
```
date
2025-11    5836
2025-12    6164
Freq: M
```

## Data types and missing values

```
TRAIN DTYPES
load_id                 object
pickup                  object
delivery                object
pickup_lat             float64
pickup_lon             float64
delivery_lat           float64
delivery_lon           float64
distance               float64
equipment               object
weight                 float64
date            datetime64[ns]
market_index           float64
quote_signal           float64
posted_rate            float64
```
```
TRAIN MISSING COUNTS
load_id           0
pickup            0
delivery          0
pickup_lat        0
pickup_lon        0
delivery_lat      0
delivery_lon      0
distance          0
equipment         0
weight          300
date              0
market_index    374
quote_signal      0
posted_rate       0
```
```
VALIDATION MISSING COUNTS
load_id           0
pickup            0
delivery          0
pickup_lat        0
pickup_lon        0
delivery_lat      0
delivery_lon      0
distance          0
equipment         0
weight          165
date              0
market_index    249
quote_signal      0
```

## Numeric distributions (train)

```
                count          mean          std          min           1%            5%          25%          50%           75%           95%           99%          max
pickup_lat    48000.0     35.647545     4.315285     28.35765    28.357650     28.756680     31.98691     35.29479     39.411040     43.438000     44.302960     44.30296
pickup_lon    48000.0    -90.928964    13.482431   -121.69849  -119.795280   -116.702490    -98.40059    -88.08915    -83.285060    -72.180510    -69.500000    -69.50000
delivery_lat  48000.0     35.641175     4.317199     28.35765    28.357650     28.756680     31.98691     35.29479     39.411040     43.438000     44.302960     44.30296
delivery_lon  48000.0    -90.857310    13.476589   -121.69849  -119.795280   -116.702490    -98.40059    -87.52871    -83.285060    -72.180510    -69.500000    -69.50000
distance      48000.0   1135.856654   728.564416     70.00000   121.899000    236.100000    550.40000    953.30000   1645.525000   2562.300000   3029.304000   3439.80000
weight        47700.0  31028.844004  9391.440620 -47500.00000  9801.960000  17490.950000  25800.00000  31436.50000  37018.000000  44934.200000  47500.000000  47500.00000
market_index  47626.0      1.083387     0.168091      0.67639     0.782887      0.836265      0.94967      1.05580      1.219590      1.365957      1.407185      1.46778
quote_signal  48000.0      2.062468     0.291391      0.69228     1.304657      1.606055      1.89103      2.05575      2.221685      2.550773      2.883951      3.61035
posted_rate   48000.0   2373.980682  1486.493245     57.22000   327.168800    599.738500   1251.55500   2030.76000   3330.750000   4953.766500   5972.834000  25533.00000
```

## Categorical distributions

- Train equipment counts:
```
equipment
Dry Van    27202
Reefer     12045
Flatbed     8753
```
- Validation equipment counts:
```
equipment
Dry Van    6780
Reefer     3051
Flatbed    2169
```
- Train top 15 pickup cities:
```
pickup
Oklahoma City    1242
Lexington        1209
Bakersfield      1193
Fort Wayne       1170
Hartford         1150
Richmond         1140
Nashville        1124
Phoenix          1121
Baton Rouge      1115
Mobile           1094
Cincinnati       1080
Shreveport       1062
Kansas City      1053
Columbia         1049
Atlanta          1029
```
- Train top 15 delivery cities:
```
delivery
Lexington        1197
Fort Wayne       1176
Baton Rouge      1167
Bakersfield      1156
Hartford         1143
Oklahoma City    1140
Richmond         1109
Atlanta          1096
Phoenix          1090
Mobile           1089
Cincinnati       1084
Columbia         1071
Nashville        1064
Tucson            997
Shreveport        983
```

## Target and relationships

- Pearson correlation with posted_rate:
```
posted_rate     1.000000
distance        0.908519
weight          0.034840
market_index    0.034165
quote_signal   -0.039858
pickup_lat     -0.090873
delivery_lat   -0.091970
pickup_lon     -0.255058
delivery_lon   -0.257086
```
- posted_rate by equipment:
```
           count         mean    median          std     min       max
equipment                                                             
Dry Van    27202  2271.548686  1953.035  1423.029914   57.22  25533.00
Flatbed     8753  2445.087223  2076.810  1505.053087  157.76  17893.17
Reefer     12045  2553.636939  2196.670  1589.670824   63.36  24140.21
```
- monthly posted_rate:
```
         count         mean    median          std
date                                              
2025-01   4918  2255.967048  1915.200  1454.274952
2025-02   4337  2273.804801  1994.250  1317.077105
2025-03   5036  2372.268092  2022.920  1462.531615
2025-04   4819  2372.162308  2044.240  1463.102457
2025-05   4913  2421.776342  2065.510  1486.071436
2025-06   4783  2497.030115  2120.220  1606.707302
2025-07   4912  2415.162030  2059.145  1504.943296
2025-08   4759  2338.407661  2015.730  1474.979672
2025-09   4670  2406.374013  2057.130  1523.483983
2025-10   4853  2379.051374  2035.900  1528.714038
```

## Potential outliers (IQR rule; descriptive only)

```
              lower_bound   upper_bound  outlier_count  outlier_pct
feature                                                            
pickup_lat      20.850715     50.547235              0     0.000000
pickup_lon    -121.073885    -60.611765            453     0.943750
delivery_lat    20.850715     50.547235              0     0.000000
delivery_lon  -121.073885    -60.611765            467     0.972917
distance     -1092.287500   3288.212500             32     0.066667
weight        8973.000000  53845.000000            424     0.888889
market_index     0.544790      1.624470              0     0.000000
quote_signal     1.395047      2.717667           1956     4.075000
posted_rate  -1867.237500   6449.542500            260     0.541667
```

## Data-quality checks

```
                            train  validation
duplicate_load_id               0         0.0
invalid_date_after_parsing      0         0.0
non_positive_distance           0         0.0
negative_weight               292       145.0
non_positive_posted_rate        0         NaN
```

## Train vs validation differences

- Columns only in train: ['posted_rate']
- Columns only in validation: []
- Validation-only pickup cities (8): ['Allentown', 'Charlotte', 'Chicago', 'Jackson', 'Knoxville', 'Laredo', 'Norfolk', 'San Diego']
- Validation-only delivery cities (8): ['Allentown', 'Charlotte', 'Chicago', 'Jackson', 'Knoxville', 'Laredo', 'Norfolk', 'San Diego']
- Validation-only equipment values (0): []
- Numeric train/validation comparison (mean and median):
```
                train_mean  validation_mean  train_median  validation_median
pickup_lat       35.647545        35.574216      35.29479           35.29479
pickup_lon      -90.928964       -90.724880     -88.08915          -87.52871
delivery_lat     35.641175        35.609538      35.29479           35.29479
delivery_lon    -90.857310       -90.913530     -87.52871          -87.52871
distance       1135.856654      1141.773100     953.30000          953.60000
weight        31028.844004     30506.636248   31436.50000        31193.00000
market_index      1.083387         0.926913       1.05580            0.92290
quote_signal      2.062468         2.051334       2.05575            2.05123
```
