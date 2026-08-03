import main as main_module
from ntc_upcoming import register_ntc_upcoming_tool


app = main_module.app
register_ntc_upcoming_tool(app, main_module)
