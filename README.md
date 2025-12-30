MAIN IDEA

Data collection occurs on Thinkpad for USRP and MiniPC for E316
Data collection happens over UHD for B200 and over IIO (Industrial Input Output framework) for E316.
→Add exact usrp.conf from Thinkpad (Pending)
Data decoding and post processing happens on my macbook where in I have the needed scripts ready to go. 

DATA COLLECTION PART

Thinkpad (Ubuntu + USRP)
soham@soham-ThinkPad-X1-Carbon-7th:~/Projects/PNT$ iridium-extractor -D 2 usrp.conf > output.bits


MiniPC (Windows + E316)
Type: Anaconda
cd C:\Users\skylo\PNT
conda activate iridium_fix
iridium-extractor -D 2 pluto.conf > output.bits

Pluto.conf
[osmosdr-source]
sample_rate=5000000
center_freq=1626276000
antenna=A_BALANCED

# REMOVE quotes. Use the URI for Ethernet connection.
device_args=uri=ip:192.168.1.10

gain=50
bandwidth=5000000



DATA PROCESSING 
Script 1. From output.bits to decoded.txt
Go to /Users/sohamdhirendesai/Desktop/Projects_2/PNT/Iridium_analysis/iridium-toolkit
Use:
 python iridium-parser.py --harder /Users/sohamdhirendesai/Desktop/Projects_2/PNT/Iridium_analysis/Data/b200_28th_night/output.bits >> /Users/sohamdhirendesai/Desktop/Projects_2/PNT/Iridium_analysis/Data/b200_28th_night/decoded.txt

Script 2. Configure and setup the position scripts
Go to /Users/sohamdhirendesai/Desktop/Projects_2/PNT/Iridium_analysis/DP_code/DP

Go to src/config/
Update: setup.py and locations.py


Copy the tle file in /Users/sohamdhirendesai/Desktop/Projects_2/PNT/Iridium_analysis/DP_code/DP/tmp

Run the optimize_start_time by:
./run.sh src/app/development/test_time_corrections.py

Go to /Users/sohamdhirendesai/Desktop/Projects_2/PNT/Iridium_analysis/
run:
./run.sh src/app/find_position_offline_2d.py

Script 3. Data Analysis
/Users/sohamdhirendesai/Desktop/Projects_2/PNT/Iridium_analysis/DP_code/DP/src/app/data_analysis
Run comprehensive_data_stats.py

Also you can run:
In DP:
Analyze_data_quality_simple.py and 
plot_differentiual_simple.py


Script 4. Differential Code
Read: DIFFERENTIAL_README.md
Run the needed code: run_differential.sh 


Script 5. Simulation Script on MATLAB
The matlab code new_plot_fixed generates csv files as a simulation files


Scripts 6. Simulation vs Experimental Data Analysis
Go to /Users/sohamdhirendesai/Desktop/Projects_2/PNT/Iridium_analysis/Comparison_matlab_vs_measured

Read FINAL_SUMMARY.md

Run:

1. `python improved_matching_with_inversion.py`
2. `python unique_matching.py`  → writes `matching_results.csv`
3. `python analyze_frequency_offset.py`
4. `python apply_corrections.py`


