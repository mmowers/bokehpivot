#This file allows reports to be run directly from this python script (without need for the bokeh server and bokehpivot UI).
#Don't edit this file directly. Make a copy (to any location), edit the copy, and run the copy as a python script in a command prompt.

#EDIT THE FOLLOWING FIELDS
bokehpivot_dir = r'\\nrelqnap02\ReEDS\bokehpivot' #path to the desired bokehpivot repo.
data_source = r'\\nrelqnap02\ReEDS\Some Project\runs\Some Runs Folder' #data_source allows all the same inputs as in the interface
scenario_filter = 'all' #'all' or string of comma-separated names.
base = 'Master' #Base case, if applicable. If base case is not needed for this report, simply leave as is.
report_path = r'\\nrelqnap02\ReEDS\Some Location\some_report.py' #Path to report that is to be run
html_num = 'one' #'one' or 'multiple'. 'one' will create one html file with all sections, and 'multiple' will create a separate html file for each section
output_dir = r'\\nrelqnap02\ReEDS\Some Location' #This is the location that will hold the report.
report_format = 'both' #'html', 'excel', or 'both'
auto_open = 'yes' #'yes' or 'no'. Automatically open the resulting report excel and html files when they are created.

#DON'T EDIT BELOW HERE
import os, sys
import importlib
import datetime
sys.path.insert(1, bokehpivot_dir)
import reeds_bokeh as rb
report_dir = os.path.dirname(report_path)
sys.path.insert(1, report_dir)
report_name = os.path.basename(report_path)[:-3]
report = importlib.import_module(report_name)
output_dir = output_dir + '/report-' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
rb.reeds_static(data_source, scenario_filter, base, report.static_presets, report_path, report_format, html_num, output_dir, auto_open)
