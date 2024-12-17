# Autolysis : An Automatic Data Analysis and Visualization Script

```
@@@@@@@@@@@@@@@%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
@@@@@@@@@@@@@@@%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
@@@@@@@@@@@@@@@@%%%%%%%#*+**%%%%%%%%%%%%%%%%%%%%%@
@@@@@@@@@@@@@@@@@@@%%+=-----.+%%%%%%%%%%%%%@@@@@@@
@@@@@@@@@@@@@@@@@@@@+%#**+*+:=#%%@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@+@@@@#@*-+#@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@*@@@@@@%=+*###%@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@%#*++#%##*+=:.:.*@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@*:.....-=...:..   .+@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@=:......:.....:...::=#@@@@@@@@@@@@@
@@@@@@@@@@@@@@@*+--:.::::::..:---=+--+%@@@@@@@@@@@
@@@@@@@@@@@@@@%-=+#+.........:-*##+----%@@@@@@@@@@
@@@@@@@@@@@@@@=::-++..........:=*%@+=-:=@@@@@@@@@@
@@@@@@@@@@@@@%:..:#+..........:-++%#-::-%@@@@@@@@@
@@@@@@@@@@@@@+  .-%=..........:=+=*@:..:%@@@@@@@@@
@@@@@@@@@@@@@:  .*%%#***********=-*@:..-@@@@@@@@@@
@@@@@@@@@@@@@: .=%%=-::-++#*-:.::.*@:.:=@@@@@@@@@@
@@@@@@@@@@@@*. -%%#::...::-:..::-=+@=.:+@@@@@@@@@@
@@@@@@@@@@@@+-=-%%+.....::-::::::=#%-::=%@@@@@@@@@
@@@@@@@%%##**++=%@=     .-:    ..+%@@%%%@@@@@@@@@@
**+=--:--===+++:#@=     :+=     .+@@@@@@@@@@@@@@@@
**#%%%@@@@%%%%%:#@=     -@+     :*@@@@@@@@@@@@@@@@
@@@@@@@@%%%%%%%:#@=     -@*     :#@@@@@@@@@@@@@@@@
@@@@@%%%%#**++=:#@#    .-@%.....:#@@@@@@@@@@@@@@@@
#*++=-:::---===*%%#.    :+@.....:*@%@@@@%%%%@@@@@@

```

## Overview
This script is designed for data analysis and visualization using Python. It leverages various libraries such as `pandas`, `matplotlib`, `seaborn`, and an API client for LLM (Large Language Model) interactions to generate insights and visualizations from datasets.

## Requirements
- Python version: >= 3.8
- Required libraries:
  - `requests >= 2.28.0`
  - `pandas >= 1.5.0`
  - `matplotlib >= 3.5.0`
  - `seaborn >= 0.12.0`
  - `numpy >= 1.21.0`
  - `rich >= 12.0.0`
  - `scipy >= 1.9.0`
  - `scikit-learn >= 1.0.0`

## Setup
1. **Install Python**: Ensure you have Python 3.8 or higher installed on your machine.
2. **Install Required Libraries**: You can install the required libraries using pip:
   ```bash
   uv pip install requests pandas matplotlib seaborn numpy rich scipy scikit-learn
   ```
3. **Set Environment Variable**: Set the `AIPROXY_TOKEN` environment variable with your API token for the LLM service. This is necessary for the script to function correctly.

   On Linux/Mac:
   ```bash
   export AIPROXY_TOKEN='your_token_here'
   ```

   On Windows:
   ```cmd
   set AIPROXY_TOKEN='your_token_here'
   ```

## Usage
To run the script, use the following command in your terminal:
```bash 
uv run autolysis *.csv
```



