import dash
from dash import dcc, html, Input, Output, State, dash_table, ctx
import dash_bootstrap_components as dbc
import pandas as pd
import joblib
from flask import send_from_directory
import os
from dotenv import load_dotenv
from datetime import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import time

# # ✅ Import DiskcacheManager from dash_extensions (not dash.long_callback!)
# from dash_extensions.enrich import DiskcacheManager, DashProxy, MultiplexerTransform
# from dash import DiskcacheManager
# import diskcache

# --- Long Callback Manager Setup ---
# This is crucial for enabling progress bars and long-running tasks.
# For simplicity, we'll use DiskcacheManager for local development.
# For production, consider CeleryManager with Redis/RabbitMQ.
# # ✅ Setup cache and manager
# cache = diskcache.Cache("./cache")
# background_callback_manager = DiskcacheManager(cache)

# Import helper functions from separate modules
from src.data_loader.file_loader import read_uploaded_file
from src.data_loader.data_cleaner import clean_game_df, clean_wallet_df, validate_file_type
from src.data_loader.feature_builder import build_player_features
from src.visuals import generate_kpi_dashboard, generate_churn_visuals, empty_figure


# Optional: Uncomment this block if you want to use Celery/Redis in production
# if 'REDIS_URL' in os.environ:
#     # Use Redis & Celery if REDIS_URL set as an env variable
#     from celery import Celery
#     celery_app = Celery(__name__, broker=os.environ['REDIS_URL'], backend=os.environ['REDIS_URL'])
#     background_callback_manager = CeleryManager(celery_app)
# else:
#     # Diskcache for non-production apps when developing locally
#     import diskcache
#     cache = diskcache.Cache("./cache")
#     background_callback_manager = DiskcacheManager(cache)


# Load environment variables for credentials
load_dotenv()

# --- Load Model and Encoders ---
# Ensure the model and encoder files are in a 'model' directory
try:
    # Assuming model is in 'model/' relative to the project root
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'churn_model.pkl')
    model = joblib.load(model_path)
    print(f"Model loaded successfully from: {model_path}")
except FileNotFoundError as e:
    print(f"Error loading model: {e}. Make sure the 'model' directory and its contents exist.")
    # Exit or handle gracefully if model files are essential for startup
    model = None # Set model to None if loading fails
except Exception as e:
    print(f"An unexpected error occurred while loading the model: {e}")
    model = None

# --- App Initialization ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True, assets_folder='assets')


# Adding small icon/logo image to title in browser's tab
# Define the path to your favicon
FAVICON_PATH = os.path.join(os.path.dirname(__file__), 'assets', 'favicon.ico') # Adjust 'logo.png' if it's actually 'favicon.ico'
# Check if the file exists before trying to serve it
if os.path.exists(FAVICON_PATH):
    # This manually registers the route for /favicon.ico
    @app.server.route('/favicon.ico')
    def serve_favicon():
        # Ensure that 'assets' is the directory containing your logo.png
        return send_from_directory(os.path.join(os.path.dirname(__file__), 'assets'), 'favicon.ico') # Serve logo.png as favicon.ico
else:
    print(f"WARNING: Favicon file not found at {FAVICON_PATH}. Favicon will not be set.")

# App title
app.title = 'NLN Churn Prediction App'
server = app.server # Expose the Flask server for production deployment




# --- Helper Functions ---
def is_valid_user(username, password):
    """Checks for correct login credentials from environment variables."""
    i = 1
    while True:
        user_key = f"APP_USER{i}"
        pass_key = f"APP_PASS{i}"
        stored_user = os.getenv(user_key)
        stored_pass = os.getenv(pass_key)
        if not stored_user:
            break
        if stored_user == username and stored_pass == password:
            return True
        i += 1
    return False

# --- User Image Mapping ---
# IMPORTANT: Replace 'Ken', 'Admin2', 'Alice', 'Bob' with the actual usernames
# you have configured in your environment variables (APP_USER1, APP_USER2, etc.)
# Make sure the image files (e.g., Ken.png, Admin2.png, default_user.png)
# are placed in your 'assets' folder.
USER_IMAGES = {
    'Ken': '/assets/Ken.png', # Example: Map username 'Ken' to 'Ken.png'
    'Admin': '/assets/Admin.png', # Example: Map username 'Admin' to 'Admin.png'
    'Alice': '/assets/Alice.png', # Female admin image
    'Bob': '/assets/Bob.png',     # Male admin image
    'John': '/assets/John.png',     # Male admin image
    'default_user': '/assets/default_user.png' # Fallback image if user has no specific mapping
}



# --- Helper Functions for Visuals/Charts ---
# Now moved to a helper module


# --- Component Definitions ---
# === Enhanced Sidebar with Submenus and Navigation Interactivity ===
# Sidebar structure with nested submenus for Home, Upload, About
sidebar = html.Div([
    html.H2([
        html.Img(src="/assets/logo.png", className="sidebar-logo"),
        html.Span("Churn Predictor", className="label")
    ], className="sidebar-title"),
    html.Hr(),

    # Vertical Navigation Pages
    # Home Section
    html.Div([
        html.Div([html.Span("🏠", className="icon"), html.Span("Home", className="label")],
                 className="menu-header", id="home-header"),
        html.Div(id="home-submenu", className="expanded-submenu menu-section", children=dbc.Nav([ # Default to expanded
            dbc.NavLink([html.Span("👋", className="icon"), html.Span("Welcome", className="label")], href="/", active="exact", className="submenu-item"),
            dbc.NavLink([html.Span("📜", className="icon"), html.Span("Wall of Fame", className="label")], href="/wall-of-fame", active="exact", className="submenu-item")
        ], vertical=True, pills=True))
    ], className="sidebar-menu-group"), # Added class for grouping in CSS

    # Upload & Predict
    html.Div([
        html.Div([html.Span("📤", className="icon"), html.Span("Upload & Predict", className="label")],
                 className="menu-header collapsible", id="upload-header"),
        html.Div(id="upload-submenu", className="collapsed-submenu menu-section", children=dbc.Nav([ # Default to collapsed
            dbc.NavLink([html.Span("📁", className="icon"), html.Span("Upload Files", className="label")], href="/upload", active="exact", className="submenu-item"),
            dbc.NavLink([html.Span("📋", className="icon"), html.Span("Prediction Table", className="label")], href="/predictions", active="exact", className="submenu-item")
        ], vertical=True, pills=True))
    ], className="sidebar-menu-group"), # Added class for grouping in CSS

    # Insights Section
    html.Div([
        html.Div([html.Span("📈", className="icon"), html.Span("Insights", className="label")],
                 className="menu-header collapsible", id="insights-header"),
        html.Div(id="insights-submenu", className="collapsed-submenu menu-section", children=dbc.Nav([ # Default to collapsed
            dbc.NavLink([html.Span("📊", className="icon"), html.Span("KPI Dashboard", className="label")], href="/kpi", active="exact", className="submenu-item"),
            dbc.NavLink([html.Span("🔮", className="icon"), html.Span("Churn Visuals", className="label")], href="/churn-visuals", active="exact", className="submenu-item")
        ], vertical=True, pills=True))
    ], className="sidebar-menu-group"), # Added class for grouping in CSS

    # About Section
    html.Div([
        html.Div([html.Span("ℹ️", className="icon"), html.Span("About", className="label")],
                 className="menu-header collapsible", id="about-header"),
        html.Div(id="about-submenu", className="collapsed-submenu menu-section", children=dbc.Nav([ # Default to collapsed
            dbc.NavLink([html.Span("🧠", className="icon"), html.Span("About App", className="label")], href="/about-app", active="exact", className="submenu-item"),
            dbc.NavLink([html.Span("👥", className="icon"), html.Span("About Team", className="label")], href="/about", active="exact", className="submenu-item"),
            dbc.NavLink([html.Span("📝", className="icon"), html.Span("Changelog", className="label")], href="/changelog", active="exact", className="submenu-item")
        ], vertical=True, pills=True))
    ], className="sidebar-menu-group"), # Added class for grouping in CSS

    # Theme Toggle
    html.Div([
        html.Div([
            html.Div([ # This div wraps the icon and label to keep them stacked above the switch
                html.Span("🌓", className="icon"),
                html.Span("Theme", className="label")
            ], className="theme-label-row"),
            dbc.Switch(
                id="theme-switch",
                label="Dark Mode",
                value=False, # Initial value will be False (light mode)
                className="theme-toggle-switch" # Custom class for styling
            )
        ], className="theme-toggle-col") # New wrapper for stacking label above switch
    ], className="theme-section"), # Overall container for the theme toggle

    html.Footer("© 2025 Lottery Analytics App. Built by Kenneth", className="sidebar-footer")
], id="sidebar", className="sidebar expanded")



# Top navbar with user dropdown and sidebar toggle
navbar = html.Div([
    html.Div(html.Button("☰", id="toggle-sidebar", className="menu-toggle"), className="navbar-left"),
    html.Div([html.Img(src='/assets/logo.png', className='logo'), html.H4("National Lottery Nigeria")], className="logo-title"),
    html.Div(id="user-info-display", className="user-panel") # This will be dynamically updated by a callback
], className="navbar")



# Main Content Area
content = html.Div(id="page-content", className="content")


# --- Page Layouts ---
login_layout = html.Div([
    html.Div([
        html.Div([
            html.Div([
                    html.Img(src="/assets/logo.png", className="login-logo"),
                    html.H1("National Lottery Nigeria", className="login-main-title"),
                    html.H3("Churn Prediction App", className="login-subtitle")
                ], className="login-title-block"),
            html.Img(src="/assets/lottery_banner.png", className="login-image")
        ], className="login-left"),
        html.Div([
            html.Div([
                html.H2("🔐 Admin Login"),
                dcc.Input(id="username", type="text", placeholder="Username", autoComplete="username", className="login-input"),
                dcc.Input(id="password", type="password", placeholder="Password", autoComplete="current-password", className="login-input"),
                html.Button("Login", id="login-button", className="login-btn"),
                html.Div(id="login-output", className="login-message")
            ], className="login-form-box")
        ], className="login-right"),
    ], className="login-wrapper")
])

home_layout = html.Div([
    html.H2("Welcome to the Online Lottery Churn Predictor", className="page-title"),
    html.P("This tool helps you identify players likely to churn based on behavioral data. Navigate using the sidebar."),
    html.P("To begin, head to the 'Upload & Predict' section to upload your CSV file and see predictions.")
], className="page-content-wrapper") # Added wrapper for consistent padding/margin

# Placeholder layouts for new pages
wall_of_fame_layout = html.Div([html.H2("Wall of Fame", className="page-title"), html.P("This is the Wall of Fame page.")], className="page-content-wrapper")

# The actual prediction table will be rendered dynamically into 'output-prediction-table-and-data'
predictions_layout = html.Div([
    html.H2("Prediction Results", className="page-title"),
    html.P("Below is the table of predicted churn for your uploaded data."),

    dbc.Row([
        dbc.Col(
            # New button to explicitly start prediction
            html.Button("▶️ Start Churn Prediction", id="start-prediction-btn", className="btn btn-success mb-3 w-100"),
            width={"size": 4, "offset": 2}, # Centered, takes 4/12 width
            className="d-flex justify-content-end" # Pushes button to the right within its column
        ),
        dbc.Col(
            html.Button("⬇ Download Predictions", id="download-btn", className="download-btn mb-3 w-100"),
            width={"size": 4}, # Takes 4/12 width
            className="d-flex justify-content-start" # Pushes button to the left within its column
        ),
    ], className="mb-4 justify-content-center"), # Center the row itself





    html.Div(id='prediction-status', className="text-center mt-2 mb-3"),
    # dcc.Store(id='prediction-status-store'),
    # Spinner wrapping both progress bar and table container
    dcc.Loading(
        id="loading-prediction",
        type="circle", # You can choose 'graph', 'cube', 'circle', 'dot', 'default'
        children=[
            # Progress bar and interval remain here, their display controlled by style
            dbc.Progress(id='prediction-progress', striped=True, animated=True, value=0, max=100, style={'marginTop': '10px', 'display': 'none'}),
            # Use simpler dcc.Interval()-based progress system.
            dcc.Interval(id='progress-interval', interval=200, n_intervals=0, disabled=True),
            # This is the container for the actual prediction table. # You can include both the progress bar and table inside the spinner if desired but usually only the final table is placed here
            html.Div(id='prediction-table-container', style={'display': 'none'}), 
        ],
        # The spinner will show whenever any of its children are being updated by callbacks
    ),

    # New navigation button to Churn Visuals
    html.Div([
        dbc.Button("🔮 Go to Churn Visuals", href="/churn-visuals", className="btn btn-info mt-4"),
    ], className="text-center")

], className="page-content-wrapper")


upload_layout = html.Div([
    html.H4("Upload Your CSV File for Prediction", className="page-title"),
    html.Div([
    html.A("📥 Download Sample Game CSV Template", href="/assets/sample_game.csv", download='sample_game.csv', className="template-btn"),
    html.Br(),
    html.A("📥 Download Sample Wallet CSV Template", href="/assets/sample_wallet.csv", download='sample_wallet.csv', className="template-btn"),
    ]),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            html.Strong("📁 Upload Game & Wallet Files: "),
            html.Div(['📤 Drag & Drop or ', html.A('Click to Upload a CSV or Excel File')]),
            html.Small("(Tip: Upload both at once or upload one now and the other later)")
            ]),
        className="upload-box",
        multiple=True # Allow multiple files upload
    ),
    # The progress bar will be used for simpler, non-interval feedback
    dbc.Progress(id='upload-progress', striped=True, animated=True, value=0, max=100, style={'marginTop': '10px', 'display': 'none'}),
    html.Div(id='upload-status', className="text-center mt-2 mb-3"), # For status messages

    # dcc.Loading now wraps the area where upload progress/summary and file previews are shown.
    # This ensures the spinner is active during the entire blocking process.
    dcc.Loading(
        id='loading-upload',
        type='circle',
        children=html.Div(id='output-data-upload') # This div will be updated by a new callback
    ),

    html.Div(id='uploaded-files-display'), # Still for the raw file names list
    html.Br(),
    html.Label("Preview Uploaded File:"),
    # dcc.Dropdown(id='preview-file-dropdown', placeholder="Select file type to preview", 
    #              options=[
    #                  {'label': 'Game', 'value': 'game'},
    #                  {'label': 'Wallet', 'value': 'wallet'}
    #                  ]),
    # html.Div(id='file-preview-container'), # This will display the preview DataTable based on dropdown
    # html.Br(),
    dcc.Dropdown(
        id='preview-file-dropdown',
        placeholder="Select file to preview", # Changed placeholder text
        # Options will be set dynamically by a callback
        options=[] # Initialize as empty
    ),
    html.Div(id='file-preview-container'), # This will display the preview DataTable
    html.Br(),
    # New navigation button to KPI Dashboard
    html.Div([
        dbc.Button("📊 Go to KPI Dashboard", href="/kpi", className="btn btn-info mt-3"),
    ], className="text-center")
], className="page-content-wrapper") # Added wrapper


# --- KPI Dashboard Layout ---
kpi_layout = html.Div([
    html.H2("📊 Key Performance Indicators (KPI) Dashboard", className="page-title"),
    html.P("Key Performance Indicators extracted from the uploaded game and wallet data."),

    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("🧮 Total Players"),
            dbc.CardBody(html.H3(id="kpi-total-players", className="card-number"))
        ], color="primary", inverse=True), width=4),

        dbc.Col(dbc.Card([
            dbc.CardHeader("🎯 Games Played"),
            dbc.CardBody(html.H3(id="kpi-games-played", className="card-number"))
        ], color="info", inverse=True), width=4),

        dbc.Col(dbc.Card([
            dbc.CardHeader("💰 Total Wallet Amount"),
            dbc.CardBody(html.H3(id="kpi-wallet-amount", className="card-number"))
        ], color="success", inverse=True), width=4)
    ], className="mb-4"),

    dcc.Loading(
        id="loading-kpi-charts",
        type="circle",
        children=[
            dcc.Graph(id='game-kpi-chart'),
            dcc.Graph(id='wallet-kpi-chart'),
            dbc.Row([
                dbc.Col(dcc.Graph(id='dau-over-time-chart'), width=6),
                dbc.Col(dcc.Graph(id='wallet-txn-by-action-channel-chart'), width=6),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='top-games-net-revenue-chart'), width=6),
                dbc.Col(dcc.Graph(id='top-games-total-plays-chart'), width=6),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='stake-prize-over-time-chart'), width=6),
                dbc.Col(dcc.Graph(id='engagement-by-hour-channel-chart'), width=6),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='access-channel-distribution-chart'), width=6),
            ]),
        ]
    ),
    
    # New navigation button back to Upload Files
    html.Div([
        dbc.Button("📁 Back to Upload Files", href="/upload", className="btn btn-secondary mt-4 me-2"),
        dbc.Button("📋 Go to Prediction Table", href="/predictions", className="btn btn-info mt-4"),
    ], className="text-center")
], className="page-content-wrapper")


# --- Churn Visuals Layout ---
churn_visuals_layout = html.Div([
    html.H2("📉 Churn Visuals", className="page-title"),
    html.P("Visualizations based on churn predictions from uploaded player data."),

    dcc.Loading(
        id="loading-churn-charts",
        type="circle",
        children=[
            dbc.Row([
                dbc.Col(dcc.Graph(id='churn-distribution-chart'), width=6),
                dbc.Col(dcc.Graph(id='feature-importance-chart'), width=6)
            ]),

            dbc.Row([
                dbc.Col(dcc.Graph(id='most-played-game-chart'), width=6),
                dbc.Col(dcc.Graph(id='stake-vs-prize-chart'), width=6)
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='days-since-last-play-churn-hist'), width=6),
                dbc.Col(dcc.Graph(id='tenure-churn-boxplot'), width=6),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='net-revenue-churn-boxplot'), width=6),
                dbc.Col(dcc.Graph(id='player-value-segment-churn-chart'), width=6),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='weekly-churn-rate-chart'), width=6),
                dbc.Col(dcc.Graph(id='reactivation-chart'), width=6),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='rfm-segment-churn-chart'), width=6),
            ]),
        ]
    ),

    # New navigation button back to Prediction Table
    html.Div([
        dbc.Button("📋 Back to Prediction Table", href="/predictions", className="btn btn-secondary mt-4"),
    ], className="text-center")
], className="page-content-wrapper")

about_app_layout = html.Div([
    html.H2("About This App", className="page-title"),
    html.P("This Dash app uses a trained machine learning model to predict churn in online lottery players. "
            "It leverages historical player data to forecast which users are at risk of churning, "
            "allowing for proactive retention strategies."),
    html.P("The application is built with Dash for interactive web dashboards, Plotly for compelling visualizations, "
            "and scikit-learn for the underlying machine learning model."),
    html.P("Built with ❤️ using Dash, Plotly, and scikit-learn.")
], className="page-content-wrapper") # Added wrapper

about_layout = html.Div([html.H2("About The Team", className="page-title"), html.P("More detailed information about the team members can be found here.")], className="page-content-wrapper")

# --- Changelog Layout ---
changelog_layout = html.Div([
    html.H2("📝 Changelog & Version History", className="page-title"),
    html.P("Below are the latest updates and release notes for the Churn Predictor App."),

    html.Ul([
        html.Li([
            html.Strong("v1.3 - July 2025"),
            html.Ul([
                html.Li("Added Churn Visuals dashboard."),
                html.Li("Improved Excel column handling and cleaning logic."),
                html.Li("Streamlined navigation structure with collapsible menus.")
            ])
        ]),
        html.Li([
            html.Strong("v1.2 - June 2025"),
            html.Ul([
                html.Li("Integrated prediction model into Dash app."),
                html.Li("Added KPI Dashboard and download functionality.")
            ])
        ]),
        html.Li([
            html.Strong("v1.0 - May 2025"),
            html.Ul([
                html.Li("Initial release with file upload, preview, and layout framework.")
            ])
        ])
    ])
], className="page-content-wrapper")



# --- Main App Layout ---
# This is the root layout of the entire application.
app.layout = html.Div([
    # dcc.Store for authentication status (True/False for logged in)
    dcc.Store(id="auth-status", storage_type="local"),
    # dcc.Store for theme preference ('light' or 'dark')
    dcc.Store(id="theme-store", storage_type="local", data={'theme': 'light'}),
    # dcc.Store to hold the current logged-in username
    dcc.Store(id="current-user", storage_type="local"),
    # dcc.Store to hold the login time
    dcc.Store(id="login-time", storage_type="local"), 
    # Will store PROCESSED game data. # Changed storage_type to 'memory' for large data to avoid browser quota limits.
    dcc.Store(id="game-store", storage_type="memory"),
    # Will store PROCESSED wallet data
    dcc.Store(id="wallet-store", storage_type="memory"),

    # Store for RAW uploaded files. This replaces your 'uploaded-files' store for raw contents
    # dcc.Store(id="uploaded-files", storage_type="memory"),
    # New store for raw uploaded files (replaces 'uploaded-files' for raw contents)
    dcc.Store(id="uploaded-raw-files-store", storage_type="memory", data={'game': None, 'wallet': None}), 
    # Store for upload/cleaning status and cleaned data for re-display
    dcc.Store(id='upload-process-status-store', data={"done": False, "error": False}, storage_type='memory'), # Renamed to avoid confusion with `upload-status` text div

     # Prediction stores (now simplified)
    dcc.Store(id='prediction-status-store', data={"done": False, "error": False}, storage_type='memory'), # Removed 'stage' and 'final_data' as it's not needed for blocking op
    dcc.Store(id='prediction-in-progress-store', data={"running": False}, storage_type='memory'), # Still useful for preventing re-clicks
    dcc.Store(id='predicted-data-store', data=None, storage_type='memory'), # Store for the final predicted data

    dcc.Store(id='preview-dropdown-store', storage_type='memory'), # New store for dropdown value
    # dcc.Download component to trigger CSV file downloads
    dcc.Download(id="download-csv"),
    # dcc.Location component to track and update the browser's URL
    dcc.Location(id="url", refresh=False),
    # This div will conditionally render the login page or the main application layout
    html.Div(id="app-container"),

    #     # Global container for prediction output and loading spinner
    # # This ensures 'output-prediction-table-and-data' always exists in the DOM
    # # but its visibility is controlled by a callback
    # html.Div(
    #     id='global-prediction-output-container',
    #     children=[
    #         dcc.Loading(
    #             id="loading-spinner",
    #             type="circle",
    #             children=html.Div(id='output-prediction-table-and-data')
    #         )
    #     ],
    #     style={'display': 'none'} # Hidden by default
    # ),
    # Dummy output for client-side callback (FIXED: Always present in layout)
    html.Div(id='dummy-output', style={'display': 'none'}) 
], id="main-container") # This ID is used for theme switching in CSS







# --- CALLBACKS ---
# Callback to render either the login page or the main application layout
@app.callback(
    Output('app-container', 'children'),
    Input('auth-status', 'data')
)
def render_app_or_login(is_logged_in):
    """Renders the main application or the login page based on auth status."""
    # If logged in, show the main app structure (sidebar + navbar + content)
    # The 'sidebar-expanded' class will control the layout behavior based on sidebar state
    if is_logged_in:
        return html.Div([
            sidebar, 
            html.Div([navbar, content], 
                     className="main-body")], id="body-wrapper", className="sidebar-expanded"
        )
    # If auth_status is None (initial load or localStorage cleared), force login page
    return login_layout



# Callback for user authentication on login button click
@app.callback(
    [Output('auth-status', 'data'),  # Updates login status
     Output('login-output', 'children'), # Displays login message (success/failure)
     Output('url', 'pathname'),
     Output('current-user', 'data'),   # Stores the logged-in username
     Output('login-time', 'data')],
    Input('login-button', 'n_clicks'),
    [State('username', 'value'),
     State('password', 'value')],
    prevent_initial_call=True
)
def authenticate(n_clicks, username, password):
    """Handles user login authentication."""
    if not username or not password:
        return dash.no_update, "Please enter username and password.", dash.no_update, dash.no_update, dash.no_update
    
    if is_valid_user(username.strip(), password.strip()):
        login_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return True, f"✅ Welcome, {username}!", "/", username.strip(), login_time
    else:
        return False, "❌ Invalid credentials.", dash.no_update, dash.no_update, dash.no_update



# Callback to handle user logout
@app.callback(
    [Output('auth-status', 'clear_data'), # Clears authentication status
     Output('current-user', 'clear_data'), # Clears stored username
     Output('predicted-data-store', 'clear_data'), # Clear stored data on logout
     Output('game-store', 'clear_data'), # Clear game data on logout
     Output('wallet-store', 'clear_data'), # Clear wallet data on logout
     Output('uploaded-raw-files-store', 'clear_data'), # Clear uploaded files on logout
     Output('preview-dropdown-store', 'clear_data'), # Clear preview dropdown selection
     Output('url', 'pathname', allow_duplicate=True)], # Redirects to login page
    Input('logout-button', 'n_clicks'),
    prevent_initial_call=True
)
def logout(n_clicks):
    if n_clicks is None: # Only proceed if a real click occurred
        # return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
        raise dash.exceptions.PreventUpdate
    # Fix for SchemaLengthValidationError: Return True for clear_data outputs
    return True, True, True, True, True, True, True, '/' # Logs out the user and Clears output-prediction data as well



# Callback to update the user dropdown menu in the navbar
@app.callback(
    Output('user-info-display', 'children'),
    [Input('current-user', 'data'),
     Input('auth-status', 'data')] # Added auth-status to determine indicator color
)
def update_user_info(username, is_logged_in):
    """Updates the navbar with user info and a logout button if a user is logged in."""
    status_color = 'green' if is_logged_in else 'red'
    user_name_display = username if username else "Guest"

    # Determine user image based on username
    user_image_src = USER_IMAGES.get(username, USER_IMAGES['default_user']) # Use .get() with a default fallback

    # Use dbc.DropdownMenu for standard dropdown behavior
    if is_logged_in:
        return dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("View Profile", disabled=True, className="user-dropdown-item"),
                dbc.DropdownMenuItem("My Attendance", disabled=True, className="user-dropdown-item"),
                dbc.DropdownMenuItem(divider=True),
                dbc.DropdownMenuItem("Logout", id="logout-button", className="logout-link user-dropdown-item"),
            ],
            nav=True,
            in_navbar=True,
            label=html.Div([
                html.Span("●", className="status-indicator", style={'color': status_color, 'margin-right': '5px'}),
                html.Img(src=user_image_src, height="24px", className="me-2 rounded-circle user-avatar-navbar"), # Added user-avatar-navbar class
                html.Span(f"Welcome, {user_name_display}")
            ], className="d-flex align-items-center"),
            className="user-dropdown-menu" # Custom class for styling the dropdown toggle and menu
        )
    else:
        # For non-logged-in users, simply show "Guest" or a simplified representation
        return html.Div([
            html.Span("●", className="status-indicator", style={'color': status_color, 'margin-right': '5px'}),
            html.Img(src=USER_IMAGES['default_user'], height="24px", className="me-2 rounded-circle user-avatar-navbar"), # Show default for guest
            html.Span("Guest")
        ], className="guest-user-panel") # Different class for guest styling


# Callback to toggle the sidebar's collapsed/expanded state
@app.callback(
    [Output('sidebar', 'className'),
     Output('body-wrapper', 'className')],
    Input('toggle-sidebar', 'n_clicks'),
    State('sidebar', 'className'),
    prevent_initial_call=True
)
def toggle_sidebar(n, current_class):
    """Toggles the sidebar between expanded and collapsed states."""
    if n:
        if 'collapsed' in (current_class or ''):
            return 'sidebar expanded', 'sidebar-expanded main-body'
        else:
            return 'sidebar collapsed', 'sidebar-collapsed main-body'
    return dash.no_update, dash.no_update



# Callback to route between different pages based on URL pathname
@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    """Displays the content for the selected page."""
    # Depending on the pathname, return the appropriate layout.
    # The global-prediction-output-container will be controlled by a separate callback.
    if pathname == '/' or pathname == '/welcome':
        return home_layout
    elif pathname == '/wall-of-fame': # New page
        return wall_of_fame_layout

    # Upload & Predict
    elif pathname == '/upload':
        return upload_layout
    elif pathname == '/predictions':
        return predictions_layout

    # Insights
    elif pathname == '/kpi':
        return kpi_layout
    elif pathname == '/churn-visuals':
        return churn_visuals_layout  # 🆕 You'll define this layout if not already

    # About
    elif pathname == '/about-app':
        return about_app_layout
    elif pathname == '/about':
        return about_layout
    elif pathname == '/changelog':
        return changelog_layout  # 🆕 You'll define this layout
    else:
        # Default to home layout if path is unrecognized or root
        return home_layout



# Callback to toggle submenu visibility using class names
@app.callback(
    [Output("home-submenu", "className"),
     Output("upload-submenu", "className"),
     Output("insights-submenu", "className"),
     Output("about-submenu", "className")],
    [Input("home-header", "n_clicks"),
     Input("upload-header", "n_clicks"),
     Input("insights-header", "n_clicks"),
     Input("about-header", "n_clicks")],
    [State("home-submenu", "className"),
     State("upload-submenu", "className"),
     State("insights-submenu", "className"),
     State("about-submenu", "className")],
    prevent_initial_call=False # Allow initial run to set classes if not already done
)
def toggle_submenu_visibility(n_home, n_upload, n_insights, n_about, home_class, upload_class, insights_class, about_class):
    """
    Toggles the visibility of submenus by changing their class names.
    This replaces the dbc.Collapse logic for custom HTML submenus.
    """
    ctx_id = ctx.triggered_id if ctx.triggered else None

    # Default to current state if no click triggered, or based on initial setup
    # Function to toggle a class between 'expanded-submenu' and 'collapsed-submenu'
    def toggle_class(current_cls):
        # Ensure 'menu-section' is always present
        if 'expanded-submenu' in current_cls:
            return "collapsed-submenu menu-section"
        elif 'collapsed-submenu' in current_cls:
            return "expanded-submenu menu-section"
        # Default to collapsed if no specific state, assuming hidden by default in CSS
        return "collapsed-submenu menu-section" # This handles initial load if classes are not set

    return (
        toggle_class(home_class) if ctx_id == "home-header" else home_class,
        toggle_class(upload_class) if ctx_id == "upload-header" else upload_class,
        toggle_class(insights_class) if ctx_id == "insights-header" else insights_class,
        toggle_class(about_class) if ctx_id == "about-header" else about_class,
    )



# FIX for Circular Dependency: Use ctx.triggered_id to prevent self-triggering loops
@app.callback(
    Output('theme-store', 'data'),
    Input('theme-switch', 'value'),
    State('theme-store', 'data'), # Get current state of the store
    prevent_initial_call=True
)
def update_store_from_switch(switch_value, current_theme_data):
    # This callback is only triggered by the user interacting with the switch.
    # We use ctx.triggered_id to confirm it was the switch that triggered this,
    # preventing a loop if theme-store.data was updated by sync_switch_with_store.
    if ctx.triggered_id == 'theme-switch':
        new_theme = 'dark' if switch_value else 'light'
        # Only update if the theme actually changed to prevent redundant updates
        if current_theme_data and current_theme_data.get('theme') == new_theme:
            raise dash.exceptions.PreventUpdate
        return {'theme': new_theme}
    raise dash.exceptions.PreventUpdate # Prevent update if not triggered by the switch


@app.callback(
    Output('theme-switch', 'value'),
    Input('theme-store', 'data'),
    prevent_initial_call=False # Run on initial load and whenever store changes
)
def sync_switch_with_store(theme_data):
    # This callback is triggered when the theme-store.data changes (either by user or initial load).
    # It sets the visual state of the switch.
    if theme_data is None:
        return False # Default to light mode for switch
    return theme_data.get('theme') == 'dark'


# The clientside callback remains the same, it listens to theme-store and applies the class to the body.
app.clientside_callback(
    """
    function(themeData) {
        if (themeData && themeData.theme === 'dark') {
            document.body.classList.add('dark-mode');
        } else {
            document.body.classList.remove('dark-mode');
        }
        return window.dash_clientside.no_update; 
    }
    """,
    # No output needed for a direct DOM manipulation
    Output('dummy-output', 'children'), # A dummy output is needed for clientside_callback
    Input('theme-store', 'data'),
    prevent_initial_call=False # Run on initial load to set theme
)


@app.callback(
    Output('main-container', 'className'),
    Input('theme-store', 'data')
)
def update_main_container_theme(theme_data):
    """Applies the theme class to the main container."""
    return theme_data.get('theme', 'light') if theme_data else 'light'




# # Callback to handle file uploads (multiple) and store cleaned data
# @app.callback(
#     [Output('output-data-upload', 'children'),
#      Output('game-store', 'data'),
#      Output('wallet-store', 'data'),
#      Output('uploaded-files', 'data'),
#      Output('upload-progress', 'value'),
#      Output('upload-progress', 'label'), # Added output for progress label
#      Output('upload-status', 'children')],
#     Input('upload-data', 'contents'),
#     State('upload-data', 'filename'),
#     prevent_initial_call=True
# )
# def handle_file_upload(contents, filenames):
#     if not contents or not filenames:
#         return html.Div("No files uploaded."), None, None, [], 0, "0%", "Waiting for upload..."

#     game_data = None
#     wallet_data = None
#     messages = []
#     uploaded_files = []

#     try:
#         total = len(filenames)
#         for idx, (content, filename) in enumerate(zip(contents, filenames), 1):
#             df = read_uploaded_file(content, filename)
#             df_type = validate_file_type(df)
#             uploaded_files.append(filename)

#             if df_type == 'game':
#                 game_df = clean_game_df(df)
#                 game_data = game_df.to_json(date_format='iso', orient='split')
#                 messages.append(html.Div(f"✅ Game file uploaded and cleaned: {filename}"))

#             elif df_type == 'wallet':
#                 wallet_df = clean_wallet_df(df)
#                 wallet_data = wallet_df.to_json(date_format='iso', orient='split')
#                 messages.append(html.Div(f"✅ Wallet file uploaded and cleaned: {filename}"))

#             else:
#                 messages.append(html.Div(f"⚠️ Could not identify file type for: {filename}"))

#         final_message = f"Uploaded {total} of {total} files. Data ready for prediction."
#         # Return all outputs at once
#         return messages, game_data, wallet_data, uploaded_files, 100, "100%", final_message
#     except Exception as e:
#         error_message = f"❌ Error processing file: {e}"
#         print(error_message)
#         return [html.Div(error_message, className="error-message")], None, None, [], 0, "0%", "Upload failed."


# # Display uploaded file names
# @app.callback(
#     Output('uploaded-files-display', 'children'),
#     Input('uploaded-files', 'data')
# )
# def display_uploaded_files(file_list):
#     if file_list:
#         return html.Ul([html.Li(f) for f in file_list])
#     return html.Div()



# --- Callbacks for Uploads Page ---
# Callback to manage the overall upload flow, processing, and status
@app.callback(
    [Output('output-data-upload', 'children'), # This is the main output for the loading spinner wrapper
     Output('game-store', 'data'),
     Output('wallet-store', 'data'),
     # Output('uploaded-files', 'data', allow_duplicate=True), # <<< REMOVED THIS LINE
     Output('upload-progress', 'value'),
     Output('upload-progress', 'label'),
     Output('upload-status', 'children'), # For user-facing messages
     Output('upload-progress', 'style'), # To show/hide the progress bar
     Output('upload-process-status-store', 'data'), # New store to track overall process status (done/error)
     Output('uploaded-raw-files-store', 'data', allow_duplicate=True) # <<< ADDED THIS TO STORE RAW CONTENTS
    ],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('uploaded-raw-files-store', 'data'), # Read current raw files
    State('game-store', 'data'), # Check if game data already loaded
    State('wallet-store', 'data'), # Check if wallet data already loaded
    State('url', 'pathname'), # To detect navigation for re-rendering
    prevent_initial_call='initial_duplicate' # Allow initial call for re-rendering on page load
)
def process_uploaded_files_and_manage_display(contents, filenames, raw_files_store, 
                                             current_game_data, current_wallet_data, pathname):
    triggered_id = ctx.triggered_id

    # Initialize all outputs with dash.no_update
    output_data_upload_children = dash.no_update
    game_data_out = dash.no_update
    wallet_data_out = dash.no_update
    # uploaded_files_out = dash.no_update # No longer an output
    progress_value_out = dash.no_update
    progress_label_out = dash.no_update
    upload_status_text_out = dash.no_update
    progress_style_out = dash.no_update
    upload_process_status_out = dash.no_update
    uploaded_raw_files_store_out = dash.no_update # New output for raw files store


    # --- Case 1: Initial page load or navigation back ---
    if triggered_id is None or (triggered_id == 'url' and pathname == '/uploads'):
        if current_game_data and current_wallet_data:
            print("DEBUG: Uploads Page: Navigated back, data already processed. Re-displaying.")
            output_data_upload_children = html.Div([
                dbc.Alert("Previously uploaded data is available. Use the dropdown below to preview.",
                          color="info", className="mt-2", fade=True, dismissable=True)
            ])
            upload_status_text_out = "✅ Data previously processed and available." # <-- Refined status message
            progress_value_out = 100
            progress_label_out = "✅ 100%"
            progress_style_out = {'marginTop': '10px', 'display': 'none'}
            upload_process_status_out = {"done": True, "error": False}
            uploaded_raw_files_store_out = raw_files_store # Ensure raw_files_store is maintained for display_uploaded_files
            # uploaded_raw_files_store_out = dash.no_update # No change to raw files store on re-render

            return output_data_upload_children, dash.no_update, dash.no_update, \
                   progress_value_out, progress_label_out, upload_status_text_out, \
                   progress_style_out, upload_process_status_out, uploaded_raw_files_store_out
        else:
            print("DEBUG: Uploads Page: Initial load or no processed data, setting default.")
            # Default initial state when no files are uploaded or processed yet
            return html.Div(), None, None, 0, "0%", "Upload your Game and Wallet files.", \
                   {'marginTop': '10px', 'display': 'none'}, {"done": False, "error": False}, {'game': None, 'wallet': None} # Reset raw files store


    # --- Case 2: Files are being uploaded ---
    if triggered_id == 'upload-data' and contents is not None and filenames is not None:
        print("DEBUG: handle_file_upload: Files uploaded, starting processing.")
        
        # Immediate feedback while blocking operation runs
        output_data_upload_children = html.Div([
            dbc.Alert("Processing your files... This may take a while for large files.",
                      color="warning", className="mt-2")
        ])
        upload_status_text_out = "Processing in progress..."
        progress_value_out = 0
        progress_label_out = "0%"
        progress_style_out = {'marginTop': '10px', 'display': 'block'}


        game_df = None
        wallet_df = None
        # processed_filenames = [] # No longer directly needed as an output of this callback

        new_raw_files_data = raw_files_store.copy() if raw_files_store else {'game': None, 'wallet': None}
        messages = [] # List to collect all status messages (including warnings)

        # Get currently active filenames from the store for duplicate check
        current_game_filename_in_store = new_raw_files_data['game']['filename'] if new_raw_files_data.get('game') else None
        current_wallet_filename_in_store = new_raw_files_data['wallet']['filename'] if new_raw_files_data.get('wallet') else None

        # Update raw_files_store with newly uploaded content
        for content, filename in zip(contents, filenames):
            normalized_filename = filename.lower()
            # --- DUPLICATE FILENAME CHECK ---
            if filename == current_game_filename_in_store or filename == current_wallet_filename_in_store:
                messages.append(html.Div(f"⚠️ Warning: File '{filename}' was already processed. Uploading again will replace it.", className="text-warning"))
                # You could add logic here to prevent upload or ask for confirmation if desired.
                # For now, we'll allow replacement but warn.

            # # Get currently active filenames from the store for duplicate check
            # # These represent the files that were *last* successfully loaded/processed
            # existing_game_filename_in_store = new_raw_files_data.get('game', {}).get('filename')
            # existing_wallet_filename_in_store = new_raw_files_data.get('wallet', {}).get('filename')

            # # Keep track of which slots are being filled by the *current* upload
            # current_upload_fills_game = False
            # current_upload_fills_wallet = False

            # # Process each newly uploaded file
            # for content, filename in zip(contents, filenames):
            #     normalized_filename = filename.lower()
                
            #     # Identify current file's intended type
            #     current_file_is_game = "game" in normalized_filename
            #     current_file_is_wallet = "wallet" in normalized_filename

            #     # --- DUPLICATE FILENAME CHECK ---
            #     # Check if this new filename is the same as the one already in the store for its type
            #     if current_file_is_game and filename == existing_game_filename_in_store:
            #         messages.append(html.Div(f"⚠️ Warning: Game file '{filename}' was already processed. Uploading again will replace it.", className="text-warning"))
            #     elif current_file_is_wallet and filename == existing_wallet_filename_in_store:
            #         messages.append(html.Div(f"⚠️ Warning: Wallet file '{filename}' was already processed. Uploading again will replace it.", className="text-warning"))
                
            #     # If the filename matches the *other* type's existing filename (unlikely but possible clash)
            #     # This check might be overly cautious or imply bad naming, but good for robustness
            #     elif current_file_is_game and filename == existing_wallet_filename_in_store:
            #         messages.append(html.Div(f"⚠️ Warning: Game file '{filename}' has same name as existing Wallet file. Please rename.", className="text-danger"))
            #     elif current_file_is_wallet and filename == existing_game_filename_in_store:
            #         messages.append(html.Div(f"⚠️ Warning: Wallet file '{filename}' has same name as existing Game file. Please rename.", className="text-danger"))

            if "game" in normalized_filename:
                new_raw_files_data['game'] = {'contents': content, 'filename': filename}
                # messages.append(html.Div(f"✅ Game file '{filename}' uploaded."))
            elif "wallet" in normalized_filename:
                new_raw_files_data['wallet'] = {'contents': content, 'filename': filename}
                # messages.append(html.Div(f"✅ Wallet file '{filename}' uploaded."))
            else:
                messages.append(html.Div(f"⚠️ Unrecognized file: '{filename}'. Please ensure filenames contain 'game' or 'wallet'.", className="text-danger"))

        # Check if both raw files are now available to proceed with cleaning
        if not new_raw_files_data.get('game') or not new_raw_files_data.get('wallet'):
            status_message_partial_text = "Waiting for both Game and Wallet files to be uploaded."
            if new_raw_files_data.get('game'): status_message_partial_text = f"Game file '{new_raw_files_data['game']['filename']}' uploaded. Waiting for Wallet file."
            elif new_raw_files_data.get('wallet'): status_message_partial_text = f"Wallet file '{new_raw_files_data['wallet']['filename']}' uploaded. Waiting for Game file."
            
            # Combine all messages for the output
            combined_messages_out = html.Div(messages + [html.P(status_message_partial_text)])

            # Return updated raw_files_store, status message, and keep everything else default/hidden
            return combined_messages_out, None, None, \
                   0, "0%", status_message_partial_text, {'marginTop': '10px', 'display': 'none'}, \
                   {"done": False, "error": False}, new_raw_files_data


        # --- Actual Blocking Processing Starts Here ---
        print("DEBUG: Starting full cleaning process for uploaded files.")
        
        try:
            # Load and Clean Game Data
            game_raw_data = new_raw_files_data['game']
            game_df_raw = read_uploaded_file(game_raw_data['contents'], game_raw_data['filename'])
            game_df = clean_game_df(game_df_raw)
            game_data_out = game_df.to_json(date_format='iso', orient='split')
            messages.append(html.Div(f"✅ Game file cleaned: {game_raw_data['filename']}"))
            print(f"DEBUG: Game data processed. Shape: {game_df.shape}")

            # Load and Clean Wallet Data
            wallet_raw_data = new_raw_files_data['wallet']
            wallet_df_raw = read_uploaded_file(wallet_raw_data['contents'], wallet_raw_data['filename'])
            wallet_df = clean_wallet_df(wallet_df_raw)
            wallet_data_out = wallet_df.to_json(date_format='iso', orient='split')
            messages.append(html.Div(f"✅ Wallet file cleaned: {wallet_raw_data['filename']}"))
            print(f"DEBUG: Wallet data processed. Shape: {wallet_df.shape}")

            final_message_text = "Data cleaning complete. Files ready for prediction."
            print("DEBUG: All files processed successfully.")

            # Combine all messages for the final output
            output_data_upload_children = html.Div(messages + [
                dbc.Alert(final_message_text, color="success", className="mt-2", fade=True, dismissable=True)
            ])
            upload_status_text_out = final_message_text
            progress_value_out = 100
            progress_label_out = "✅ 100%"
            progress_style_out = {'marginTop': '10px', 'display': 'none'}
            upload_process_status_out = {"done": True, "error": False}
            
            # The uploaded-raw-files-store will also be updated with the final processed content
            # This ensures that `display_uploaded_files` shows the correct names.
            uploaded_raw_files_store_out = new_raw_files_data


            return output_data_upload_children, game_data_out, wallet_data_out, \
                   progress_value_out, progress_label_out, upload_status_text_out, \
                   progress_style_out, upload_process_status_out, uploaded_raw_files_store_out
        
        except Exception as e:
            error_message_text = f"❌ Error processing files: {e}"
            print(error_message_text)
            import traceback
            traceback.print_exc()
            
            # Combine all messages for the error output
            output_data_upload_children = html.Div(messages + [
                dbc.Alert(error_message_text, color="danger", className="mt-2", fade=True, dismissable=True)
            ])
            upload_status_text_out = "Upload failed. " + str(e)
            progress_value_out = 0
            progress_label_out = "0%"
            progress_style_out = {'marginTop': '10px', 'display': 'none'}
            upload_process_status_out = {"done": False, "error": True}
            
            # Ensure raw files store is cleared or retained based on desired behavior on error
            uploaded_raw_files_store_out = {'game': None, 'wallet': None} # Clear raw files on error
            
            return output_data_upload_children, None, None, \
                   progress_value_out, progress_label_out, upload_status_text_out, \
                   progress_style_out, upload_process_status_out, uploaded_raw_files_store_out
    
    raise dash.exceptions.PreventUpdate

# Display uploaded file names
@app.callback(
    Output('uploaded-files-display', 'children'),
    Input('uploaded-raw-files-store', 'data') # NOW READS FROM uploaded-raw-files-store
)
def display_uploaded_files(raw_files_data):
    if raw_files_data:
        file_names = []
        if raw_files_data.get('game') and raw_files_data['game'].get('filename'):
            file_names.append(raw_files_data['game']['filename'])
        if raw_files_data.get('wallet') and raw_files_data['wallet'].get('filename'):
            file_names.append(raw_files_data['wallet']['filename'])
        
        if file_names:
            return html.Div([
                html.P("Currently Uploaded Raw Files:"), # Adjusted text for clarity
                html.Ul([html.Li(f) for f in file_names])
            ])
    return html.Div()



# NEW CALLBACK: To dynamically update the dropdown options
@app.callback(
    Output('preview-file-dropdown', 'options'),
    Input('upload-process-status-store', 'data'), # Triggered when cleaning status changes
    State('uploaded-raw-files-store', 'data'), # Get raw filenames
    # ADDED Input('url', 'pathname') to trigger on page load/navigation
    Input('url', 'pathname'),
    # ADDED prevent_initial_call='initial_duplicate' to allow initial call
    # This ensures it fires when the page loads or you navigate back
    prevent_initial_call='initial_duplicate'
)
def update_preview_dropdown_options(upload_process_status, uploaded_raw_files_data, pathname):
    # Ensure this only runs if we are on the /upload page
    if pathname != "/upload":
        raise dash.exceptions.PreventUpdate

    options = []
    # Only populate options if cleaning is done AND there's no error AND we have raw file data
    # This also covers the case where navigating back to a 'done' state.
    if upload_process_status.get("done") and not upload_process_status.get("error"):
        if uploaded_raw_files_data: # Ensure uploaded_raw_files_data is not None
            if uploaded_raw_files_data.get('game') and uploaded_raw_files_data['game'].get('filename'):
                # The 'label' is what the user sees, 'value' is what the callback receives
                options.append({'label': uploaded_raw_files_data['game']['filename'], 'value': 'game'})
            if uploaded_raw_files_data.get('wallet') and uploaded_raw_files_data['wallet'].get('filename'):
                options.append({'label': uploaded_raw_files_data['wallet']['filename'], 'value': 'wallet'})
    
    print(f"DEBUG: update_preview_dropdown_options: Options generated: {options} on path {pathname}")
    return options


# Callback to save dropdown value to store (no change needed)
@app.callback(
    Output('preview-dropdown-store', 'data'),
    Input('preview-file-dropdown', 'value'),
    prevent_initial_call=True
)
def save_preview_dropdown_value(value):
    return value

# Callback to load dropdown value from store (will use dynamic changing 'game'/'wallet' value)
@app.callback(
    Output('preview-file-dropdown', 'value'),
    Input('preview-dropdown-store', 'data'),
    Input('url', 'pathname'),
    prevent_initial_call='initial_duplicate' # Allow initial call for persistence
)
def load_preview_dropdown_value(stored_value, pathname):
    if pathname != "/uploads":
        raise dash.exceptions.PreventUpdate
    return stored_value


# Preview selected uploaded file
@app.callback(
    Output('file-preview-container', 'children'),
    Input('preview-file-dropdown', 'value'),
    State('game-store', 'data'),
    State('wallet-store', 'data'),
    State('upload-process-status-store', 'data'), # Check overall process status
    prevent_initial_call='initial_duplicate' # Allow initial call for persistence
)
def preview_uploaded_file(selected_value, game_data, wallet_data, upload_process_status_store):
    # 'selected_value' will be 'game' or 'wallet' as set in the options callback

    # Only try to preview if cleaning is done and a file type is selected
    if not upload_process_status_store.get("done") or not selected_value:
        return html.Div("Upload and process your files, then select a file from the dropdown to preview data.", className="text-muted p-3")

    df = pd.DataFrame() # Initialize empty DataFrame
    if selected_value == 'game' and game_data:
        try:
            df = pd.read_json(game_data, orient='split')
        except Exception as e:
            return html.Div(f"Error loading Game data for preview: {e}", className="error-message p-3")
    elif selected_value == 'wallet' and wallet_data:
        try:
            df = pd.read_json(wallet_data, orient='split')
        except Exception as e:
            return html.Div(f"Error loading Wallet data for preview: {e}", className="error-message p-3")
    else:
        return html.Div(f"No {selected_value.capitalize()} data available or processed yet.", className="text-muted p-3")

    if df.empty:
        return html.Div(f"Selected file type '{selected_value}' has no data to display or data is empty after cleaning.", className="text-muted p-3")
    
    return dash_table.DataTable(
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.head(10).to_dict('records'), # Preview first 10 rows
        page_size=10,
        style_table={'overflowX': 'auto', 'minWidth': '100%'},
        style_cell={
            'textAlign': 'left', 'fontFamily': 'Nunito', 'padding': '8px',
            'border': '1px solid var(--dash-cell-border-light)', 'whiteSpace': 'normal',
            'height': 'auto', 'minWidth': '100px', 'width': '120px', 'maxWidth': '180px',
        },
        style_header={
            'backgroundColor': 'var(--dash-header-bg-light)', 'color': 'var(--dash-header-text-light)',
            'fontWeight': 'bold', 'border': '1px solid var(--dash-header-bg-light)'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'var(--dash-cell-odd-bg-light)'
            },
            { # This was likely causing dark mode issues if var(--dash-cell-bg-dark) wasn't defined
                'if': {'column_id': 'any'},
                'backgroundColor': 'var(--dash-cell-bg-dark)',
                'color': 'var(--dash-cell-text-dark)',
                'border': '1px solid var(--dash-cell-border-dark)'
            }
        ],
        css=[{
            'selector': '.dash-header',
            'rule': 'background-color: var(--dash-header-bg-dark) !important; color: var(--dash-header-text-dark) !important;'
        }]
    )



# Callback to update kpi dashboard page visuals/charts
@app.callback(
    [Output("kpi-total-players", "children"),
     Output("kpi-games-played", "children"),
     Output("kpi-wallet-amount", "children"),
     Output("game-kpi-chart", "figure"),
     Output("wallet-kpi-chart", "figure"),
     Output("dau-over-time-chart", "figure"), # New output
     Output("wallet-txn-by-action-channel-chart", "figure"), # New output
     Output("top-games-net-revenue-chart", "figure"), # New output
     Output("top-games-total-plays-chart", "figure"), # New output
     Output("stake-prize-over-time-chart", "figure"), # New output
     Output("engagement-by-hour-channel-chart", "figure"), # New output
     Output("access-channel-distribution-chart", "figure")], # New output
    [Input("wallet-store", "data"),
     Input("game-store", "data"),
     Input("theme-store", "data")] # Changed to Input to react to theme changes
)
def update_kpi_dashboard(wallet_data, game_data, theme_store):
    theme = theme_store.get('theme', 'light')

    if not wallet_data or not game_data:
        #         return "0", "0", "₦0.00", go.Figure(), go.Figure()
        # Return all outputs as empty figures if data is missing
        return ("0", "0", "₦0.00",
                empty_figure("Game Chart (No Data)", theme),
                empty_figure("Wallet Chart (No Data)", theme),
                empty_figure("Daily Active Users Over Time (No Data)", theme),
                empty_figure("Wallet Transactions by Action & Channel (No Data)", theme),
                empty_figure("Top 10 Games by Net Revenue (No Data)", theme),
                empty_figure("Top 10 Games by Total Plays (No Data)", theme),
                empty_figure("Daily Total Stake vs. Prize Over Time (No Data)", theme),
                empty_figure("Player Engagement by Hour of Day and Access Channel (No Data)", theme),
                empty_figure("Access Channel Distribution (No Data)", theme))

    try:
        wallet_df = pd.read_json(wallet_data, orient='split')
        game_df = pd.read_json(game_data, orient='split')

        kpis = generate_kpi_dashboard(game_df, wallet_df, theme)
        return (
            kpis['total_players'],
            kpis['total_games_played'],
            kpis['total_wallet_amount'],
            kpis['game_fig'],
            kpis['wallet_fig'],
            kpis['dau_chart'], # New return
            kpis['wallet_txn_by_action_channel_chart'], # New return
            kpis['top_games_net_revenue_chart'], # New return
            kpis['top_games_total_plays_chart'], # New return
            kpis['stake_prize_over_time_chart'], # New return
            kpis['engagement_by_hour_channel_chart'], # New return
            kpis['access_channel_distribution_chart'] # New return
        )
    except Exception as e:
        print(f"❌ KPI update error: {e}")
        # Return all outputs as empty figures on error
        return ("0", "0", "₦0.00",
                empty_figure(f"Game Chart Error: {e}", theme),
                empty_figure(f"Wallet Chart Error: {e}", theme),
                empty_figure(f"DAU Chart Error: {e}", theme),
                empty_figure(f"Wallet Txn Chart Error: {e}", theme),
                empty_figure(f"Top Games Revenue Chart Error: {e}", theme),
                empty_figure(f"Top Games Plays Chart Error: {e}", theme),
                empty_figure(f"Stake/Prize Chart Error: {e}", theme),
                empty_figure(f"Engagement Chart Error: {e}", theme),
                empty_figure(f"Access Channel Dist. Error: {e}", theme))



# Callback to update churn visuals page visuals/charts
@app.callback(
    [Output("churn-distribution-chart", "figure"),
     Output("feature-importance-chart", "figure"),
     Output("most-played-game-chart", "figure"),
     Output("stake-vs-prize-chart", "figure"),
     Output("days-since-last-play-churn-hist", "figure"), # New output
     Output("tenure-churn-boxplot", "figure"), # New output
     Output("net-revenue-churn-boxplot", "figure"),
     Output("player-value-segment-churn-chart", "figure"), # New output
     Output("weekly-churn-rate-chart", "figure"), # New output
     Output("reactivation-chart", "figure"), # New output
     Output("rfm-segment-churn-chart", "figure")], # New output
    [Input("predicted-data-store", "data"),
     Input("theme-store", "data")]  # CHANGED from State to Input
)
def update_churn_visuals(predicted_data, theme_store):
    theme = theme_store.get('theme', 'light')
    try:
        df = pd.read_json(predicted_data, orient='split') if predicted_data else pd.DataFrame()
    except Exception as e:
        print(f"[Churn Visuals] DataFrame conversion error: {e}")
        # return go.Figure(), go.Figure(), go.Figure(), go.Figure()
        return (empty_figure("Churn Distribution (Error)", theme),
                empty_figure("Feature Importance (Error)", theme),
                empty_figure("Most Played Game by Churn (Error)", theme),
                empty_figure("Stake vs Prize (Error)", theme),
                empty_figure("Days Since Last Play vs Churn (Error)", theme), # New return
                empty_figure("Player Tenure vs Churn (Error)", theme), # New return
                empty_figure("Net Revenue vs Churn (Error)", theme), # New return
                empty_figure("Churn Distribution by Value Segment (Error)", theme), # New return
                empty_figure("Weekly Churn Rate (Error)", theme), # New return
                empty_figure("Player Reactivation After Win (Error)", theme), # New return
                empty_figure("RFM Segment Churn Analysis (Error)", theme)) # New return

    visuals = generate_churn_visuals(df, model, theme)

    return (
        visuals.get('churn-distribution-chart', empty_figure("Churn Distribution (No Data)")),
        visuals.get('feature-importance-chart', empty_figure("Feature Importance (No Data)")),
        visuals.get('most-played-game-chart', empty_figure("Most Played Game by Churn (No Data)")),
        visuals.get('stake-vs-prize-chart', empty_figure("Stake vs Prize (No Data)")),
        visuals.get('days-since-last-play-churn-hist', empty_figure("Days Since Last Play vs Churn (No Data)")), # New return
        visuals.get('tenure-churn-boxplot', empty_figure("Player Tenure vs Churn (No Data)")), # New return
        visuals.get('net-revenue-churn-boxplot', empty_figure("Net Revenue vs Churn (No Data)")), # New return
        visuals.get('player-value-segment-churn-chart', empty_figure("Churn Distribution by Player Value Segmentation (No Data)")), # New return
        visuals.get('weekly-churn-rate-chart', empty_figure("Weekly Churn Rate (No Data)")), # New return
        visuals.get('reactivation-chart', empty_figure("Player Reactivation After Win (No Data)")), # New return
        visuals.get('rfm-segment-churn-chart', empty_figure("RFM Segment Churn Analysis (No Data)")) # New return
    )



# --- Prediction Page Callbacks (REFINED for re-rendering fix) ---
# Callback to combine, build features, and store predictions
@app.callback(
    [Output('predicted-data-store', 'data'),
     Output('prediction-status-store', 'data'),
     Output('prediction-status', 'children'),
     Output('prediction-progress', 'value'),
     Output('prediction-progress', 'label'),
     Output('prediction-progress', 'style'),
     Output('prediction-in-progress-store', 'data') 
    ],
    Input('start-prediction-btn', 'n_clicks'),
    State('game-store', 'data'),
    State('wallet-store', 'data'),
    State('prediction-status-store', 'data'),
    State('predicted-data-store', 'data'), # Current predicted data for re-rendering check
    State('prediction-in-progress-store', 'data'), # Current running state
    State('url', 'pathname'), # To detect navigation for re-rendering
    prevent_initial_call='initial_duplicate' 
)
def run_prediction(n_clicks, game_data, wallet_data, status_store, predicted_data_current, in_progress_store, pathname):
    triggered_id = ctx.triggered_id

    # Initialize all outputs with dash.no_update
    predicted_data_out = dash.no_update
    prediction_status_store_out = dash.no_update
    prediction_status_text_out = dash.no_update
    progress_value_out = dash.no_update
    progress_label_out = dash.no_update
    progress_style_out = dash.no_update
    in_progress_store_out = dash.no_update

    # --- Case 1: Initial page load or navigation back ---
    if triggered_id is None or (triggered_id == 'url' and pathname == '/predictions'):
        if status_store.get("done") and predicted_data_current:
            print(f"DEBUG: {datetime.now()} Prediction Page: Navigated back, data already predicted. Re-displaying.")
            
            # Key Change: Explicitly return predicted_data_current to re-trigger render_prediction_table
            predicted_data_out = predicted_data_current 
            
            prediction_status_text_out = "✅ Previous prediction available."
            progress_value_out = 100
            progress_label_out = "✅ 100%"
            progress_style_out = {'marginTop': '10px', 'display': 'none'} # Hide progress bar
            prediction_status_store_out = {"done": True, "error": False}
            in_progress_store_out = {"running": False} # Ensure not running
            
            return predicted_data_out, prediction_status_store_out, prediction_status_text_out, \
                   progress_value_out, progress_label_out, progress_style_out, in_progress_store_out
        else:
            print(f"DEBUG: {datetime.now()} Prediction Page: Initial load or no predicted data, setting default.")
            return None, {"done": False, "error": False}, "Click 'Start Churn Prediction' to begin.", \
                   0, "0%", {'marginTop': '10px', 'display': 'none'}, {"running": False}


    # --- Case 2: 'Start Prediction' button clicked ---
    if triggered_id == 'start-prediction-btn' and n_clicks is not None:
        if in_progress_store and in_progress_store.get("running"):
            print(f"DEBUG: {datetime.now()} Prediction already in progress, ignoring button click.")
            return dash.no_update, dash.no_update, "🔄 Prediction already in progress...", \
                   dash.no_update, dash.no_update, dash.no_update, dash.no_update 

        if not game_data or not wallet_data:
            print(f"DEBUG: {datetime.now()} Missing Game/Wallet data for prediction.")
            return None, {"done": False, "error": True}, "❌ Please upload Game & Wallet files first.", \
                   0, "0%", {'marginTop': '10px', 'display': 'none'}, {"running": False}

        # If prediction already completed for current data (simple check)
        if status_store.get("done") and predicted_data_current:
            print(f"DEBUG: {datetime.now()} Prediction already complete, re-displaying previous results.")
            # Key Change: Explicitly return predicted_data_current to re-trigger render_prediction_table
            predicted_data_out = predicted_data_current 
            
            return predicted_data_out, status_store, "✅ Prediction already complete. Displaying previous results.", \
                   100, "✅ 100%", {'marginTop': '10px', 'display': 'none'}, {"running": False}

        # --- Actual Blocking Prediction Process Starts Here ---
        print(f"DEBUG: {datetime.now()} Starting full prediction process.")
        
        predicted_data_out = None # Clear previous data if re-running
        prediction_status_text_out = "🔄 Running prediction... This may take a while."
        progress_value_out = 0 
        progress_label_out = "0%"
        progress_style_out = {'marginTop': '10px', 'display': 'block'} 
        in_progress_store_out = {"running": True} 

        try:
            print(f"DEBUG: {datetime.now()} Prediction: Loading game_df...")
            game_df = pd.read_json(game_data, orient='split')
            print(f"DEBUG: {datetime.now()} Prediction: Loading wallet_df...")
            wallet_df = pd.read_json(wallet_data, orient='split')
            print(f"DEBUG: {datetime.now()} Prediction: DataFrames loaded.")

            print(f"DEBUG: {datetime.now()} Prediction: Building player features...")
            player_features = build_player_features(game_df, wallet_df)
            print(f"DEBUG: {datetime.now()} Prediction: Player features built.")

            expected_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else \
                                player_features.drop(columns=['player_id'], errors='ignore').columns

            X = player_features[player_features.columns.intersection(expected_features)].copy()
            X = X.reindex(columns=expected_features, fill_value=0)
            X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
            print(f"DEBUG: {datetime.now()} Prediction: Features prepared. X shape: {X.shape}")

            print(f"DEBUG: {datetime.now()} Prediction: Running model.predict...")
            player_features['prediction'] = model.predict(X)
            print(f"DEBUG: {datetime.now()} Prediction: Running model.predict_proba...")
            player_features['prediction_proba'] = model.predict_proba(X)[:, 1]
            print(f"DEBUG: {datetime.now()} Prediction: Predictions made.")

            final_data = player_features.to_json(date_format='iso', orient='split')
            print(f"DEBUG: {datetime.now()} Prediction: Final data serialized.")

            predicted_data_out = final_data
            prediction_status_store_out = {"done": True, "error": False}
            prediction_status_text_out = "✅ Prediction complete!"
            progress_value_out = 100
            progress_label_out = "✅ 100%"
            progress_style_out = {'marginTop': '10px', 'display': 'none'} 
            in_progress_store_out = {"running": False} 

            return predicted_data_out, prediction_status_store_out, prediction_status_text_out, \
                   progress_value_out, progress_label_out, progress_style_out, in_progress_store_out

        except Exception as e:
            print(f"Prediction Error: {e}")
            import traceback
            traceback.print_exc()
            predicted_data_out = None
            prediction_status_store_out = {"done": False, "error": True}
            prediction_status_text_out = f"❌ Error during prediction: {e}"
            progress_value_out = 0
            progress_label_out = "0%"
            progress_style_out = {'marginTop': '10px', 'display': 'none'}
            in_progress_store_out = {"running": False} 

            return predicted_data_out, prediction_status_store_out, prediction_status_text_out, \
                   progress_value_out, progress_label_out, progress_style_out, in_progress_store_out
    
    raise dash.exceptions.PreventUpdate



# Callback to render the prediction table
@app.callback(
    Output('prediction-table-container', 'children'),
    Output('prediction-table-container', 'style'),
    Input('predicted-data-store', 'data'),
    State('prediction-status-store', 'data'),
    prevent_initial_call='initial_duplicate' # Allow initial call for persistence
)
def render_prediction_table(predicted_data, status_store):
    # Only render if prediction is done and data exists
    if not status_store.get("done") or not predicted_data:
        return None, {'display': 'none'}
    
    try:
        df = pd.read_json(predicted_data, orient='split')
        columns = [
            {"name": col.replace('_', ' ').title(), "id": col}
            for col in df.columns if col not in ['churned', 'churn_likelihood_score', 'likely_to_churn']
        ]
        style_data_conditional = [
            {
                'if': {'column_id': 'prediction', 'filter_query': '{prediction} eq 1'},
                'backgroundColor': '#ffe0e0',
                'color': '#dc3545',
                'fontWeight': 'bold'
            },
            {
                'if': {'column_id': 'prediction', 'filter_query': '{prediction} eq 0'},
                'backgroundColor': '#e0ffe0',
                'color': '#28a745',
                'fontWeight': 'bold'
            }
        ]
        table = html.Div([
            html.H5("Predicted Churn Data", className="mt-4 mb-3"),
            dash_table.DataTable(
                id='prediction-table',
                columns=columns,
                data=df.to_dict('records'),
                page_size=15,
                sort_action='native',
                filter_action='native',
                style_table={'overflowX': 'auto', 'minWidth': '100%'},
                style_cell={
                    'textAlign': 'left', 'fontFamily': 'Nunito', 'padding': '10px',
                    'border': '1px solid var(--dash-cell-border-light)', 'whiteSpace': 'normal',
                    'height': 'auto', 'minWidth': '100px', 'width': '120px', 'maxWidth': '180px',
                },
                style_header={
                    'backgroundColor': 'var(--dash-header-bg-light)', 'color': 'var(--dash-header-text-light)',
                    'fontWeight': 'bold', 'border': '1px solid var(--dash-header-bg-light)'
                },
                style_data_conditional=style_data_conditional
            ),
            dbc.Alert("Note: 'prediction' = 1 for churn, 0 for no churn. 'prediction_proba' shows likelihood/probability of churn.",
                      color="info", className="mt-2", fade=True, dismissable=True)
        ], className="dash-spreadsheet")
        return table, {'display': 'block'}
    except Exception as e:
        print(f"Error rendering prediction table: {e}")
        return html.Div(f"Error rendering prediction table: {e}", className="error-message"), {'display': 'block'}


# Trigger CSV download
@app.callback(
    Output('download-csv', 'data'),
    Input('download-btn', 'n_clicks'),
    State('predicted-data-store', 'data'),
    prevent_initial_call=True
)
def download_predictions(n_clicks, stored_data):
    """Triggers the download of the prediction results."""
    if n_clicks and stored_data: # Ensure n_clicks is not None and stored_data exists
        print(f"Download button clicked {n_clicks} times. Preparing download...")
        df = pd.read_json(stored_data, orient='split')
        return dcc.send_data_frame(df.to_csv, "churn_predictions.csv", index=False)
    return dash.no_update

# Mistake made when pushing

if __name__ == '__main__':
    app.run(debug=True)

# server = app.server  # For Render
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8080)
