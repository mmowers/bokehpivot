import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import reeds_bokeh as rb
import importlib
import datetime

data_source = '//nrelqnap02/ReEDS/Some Project/runs/Some Runs Folder'
scenario_filter = 'all' #'all' or comma separated string
base = 'Master'
report_path = '//nrelqnap02/ReEDS/Some Location/report.py'
html_num = 'one' #'one' or 'multiple'
output_dir = '//nrelqnap02/ReEDS/Some Location/report-' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
auto_open = 'yes'

report_dir = os.path.dirname(report_path)
sys.path.insert(1, report_dir)
report_name = os.path.basename(report_path)[:-3]
report = importlib.import_module(report_name)
rb.reeds_static(data_source, scenario_filter, base, report.static_presets, report_path, 'both', html_num, output_dir, auto_open)
