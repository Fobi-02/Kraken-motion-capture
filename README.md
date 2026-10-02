##  Set-Up
git clone <repo-url>
cd Kraken-motion-capture

python3 -m venv .venv
source .venv/bin/activate
pip install -e .

After that, every time they open a new terminal, they only need:

source .venv/bin/activate
python3 -m data_analysis.main # Python treats data_analysis as a package and main as a module inside it.
python3 data_analysis/main.py # Python treats main.py as the directly executed script (same but may not work for big projects)