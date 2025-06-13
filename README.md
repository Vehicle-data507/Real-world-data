#******************************** Real-world-data information******************************************#

#Data acquisition method:
The vehicle onboard sensor signals, including temperature, current, voltage, and vehicle-level signals, are transmitted to the Battery Management System (BMS) via the Controller Area Network (CAN) bus. Subsequently, 
the data are forwarded by the onboard telematics unit (T-BOX) to the signal base station at a fixed sampling frequency. The base station then transmits the collected data to the Original Equipment Manufacturer (OEM) and the big data monitoring platform for storage. 
The publicly available dataset utilized in this study includes the following variables: data sampling timestamp, vehicle speed, charging status, accumulated mileage, total voltage, current, state of charge (SOC), and cell-level parameters such as minimum and maximum cell voltages and temperatures.

#******************************Description of dataset variables******************************************#

Time: Timestamp of data acquisition.

vhc_speed: Vehicle speed (km/h).

charging_signal: Charging status indicator; a value of 3 denotes driving mode, while 1 indicates charging mode.

vhc_totalMile: Accumulated driving mileage (km).

hv_voltage: Total voltage of the battery pack (V).

hv_current: Total current of the battery pack (A).

bcell_soc: State of Charge (SOC) of the battery pack.

bcell_maxVoltage: Maximum cell voltage within the battery pack (V).

bcell_minVoltage: Minimum cell voltage within the battery pack (V).

bcell_maxTemp: Maximum cell temperature within the battery pack (°C).

bcell_minTemp: Minimum cell temperature within the battery pack (°C).


#*****************************Vehicle details form************************************************#

| Vehicle number | Vehicle type     | Battery material | Initial rated capacity (Ah) | Number of data points | Cumulative Mileage (km) | Sampling frequency (Hz) |
|----------------|------------------|-------------------|-----------------------------|-----------------------|-------------------------|-------------------------|
| Vehicle#1      | Passenger vehicle | NCM              | 150                         | 954754                | 69043                   | 0.1                     |
| Vehicle#2      | Passenger vehicle | NCM              | 150                         | 998243                | 73950                   | 0.1                     |
| Vehicle#3      | Passenger vehicle | NCM              | 160                         | 997098                | 79440                   | 0.1                     |
| Vehicle#4      | Passenger vehicle | NCM              | 160                         | 1150999               | 96279                   | 0.1                     |
| Vehicle#5      | Passenger vehicle | NCM              | 160                         | 1096073               | 114413                  | 0.1                     |
| Vehicle#6      | Passenger vehicle | NCM              | 160                         | 501031                | 27318                   | 0.1                     |
| Vehicle#7      | Passenger vehicle | LFP              | 120                         | 5304111               | 32496                   | 0.5                     |
| Vehicle#8      | Electric bus      | LFP              | 645                         | 675236                | 82668                   | 0.1                     |
| Vehicle#9      | Electric bus      | LFP              | 645                         | 443806                | 43988                   | 0.1                     |
| Vehicle#10     | Electric bus      | LFP              | 505                         | 715956                | 27677                   | 0.1                     |


#*****************************Battery pack structure table************************************************#

Battery pack structure: Owing to confidentiality constraints imposed by the data platform and OEM manufacturers, the structural details of some vehicle battery packs remain undisclosed. Consequently, 
we have released only the portions of information that were accessible under the current data-sharing agreements.
| Vehicle number | Battery pack structure                  |
|----------------|-------------------------------------------|
| Vehicle#1      | 91 battery cells connected in series     |
| Vehicle#2      | 91 battery cells connected in series     |
| Vehicle#3      | 91 battery cells connected in series     |
| Vehicle#4      | 91 battery cells connected in series     |
| Vehicle#5      | 91 battery cells connected in series     |
| Vehicle#6      | 91 battery cells connected in series     |
| Vehicle#7      | ——                                       |
| Vehicle#8      | ——                                       |
| Vehicle#9      | It contains a total of 360 battery cells |
| Vehicle#10     | It contains a total of 324 battery cells |


#****************************Environment************************************************#
python=3.8.20
pandas=2.0.3
numpy=1.24.1
matplotlib=3.7.5
scikit-learn=1.3.2
seaborn=0.13.2

#****************************Code explanation************************************************#
Data preprocessing code, where the parameter identification code is 17.1 Parameter Identification.ipynb

1_Outlier detection.ipynb

2_Check whether the deleted column names are consistent Outlier detection.ipynb

3_Extraction date.ipynb

4_Check whether the date and mileage are increasing Extract date.ipynb

5_Divided into multiple documents by month.ipynb

6_Check whether the third column is all 1 or 3_charging status.ipynb

7_Combination date.ipynb

8_Matching ambient temperature.ipynb

9_Calculate the cumulative number of days.ipynb

10_Calculate cumulative charge and discharge capacity.ipynb

11_Calculate power.ipynb

12_Calculate battery temperature.ipynb

13_Split segments.ipynb

14_Calculate the average current, average temperature, and average power.ipynb

15_Statistics on cumulative deep discharge charging and fast charging times.ipynb

16_interpolation.ipynb

17.1 Parameter Identification.ipynb

17.2 Noise Reduction and Smoothing.ipynb

17.3 Additions to the original document.ipynb

18_Calculate SOH.ipynb

19_Noise reduction, smoothing.ipynb

20_Rename.ipynb

21_SOH_2.ipynb

Baseline model code
Transformer_dataset_2.ipynb
CNN_dataset_2.ipynb
LSTM_dataset_2.ipynb





