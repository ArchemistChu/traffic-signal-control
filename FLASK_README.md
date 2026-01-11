# Flask Web Application

This is the Flask version of the Traffic Signal Control System, converted from Streamlit.

## Running the Flask Application

### Option 1: Direct Python execution
```bash
python app_flask.py
```

### Option 2: Using Flask CLI
```bash
flask --app app_flask run
```

The application will be available at: `http://localhost:5000`

## Features

- **Dataset Selection**: Choose between Custom Single Intersection, 4-Intersection Grid, or TAPASCologne
- **Strategy Selection**: Fixed-time, Adaptive, DQN, or MaxPressure control
- **Real-time Simulation**: Launch SUMO simulations with GUI
- **Results Visualization**: View metrics, charts, and comparisons
- **History Management**: Track multiple strategy runs and compare performance

## Differences from Streamlit Version

1. **Web Framework**: Uses Flask instead of Streamlit
2. **Templates**: HTML templates in `templates/` directory
3. **Static Files**: CSS/JS in `static/` directory (if needed)
4. **Session Management**: Uses Flask sessions instead of Streamlit session state
5. **AJAX Updates**: Uses JavaScript/jQuery for real-time updates instead of Streamlit's auto-refresh

## File Structure

```
.
├── app_flask.py          # Main Flask application
├── templates/
│   ├── base.html         # Base template with common layout
│   └── index.html        # Main page template
├── static/               # Static files (CSS, JS, images)
└── requirements.txt      # Updated with Flask dependency
```

## API Endpoints

- `GET /` - Main page
- `POST /set_dataset` - Change dataset
- `POST /start_simulation` - Start simulation
- `GET /check_simulation` - Check simulation status
- `POST /clear_history` - Clear simulation history
- `GET /get_comparison_charts` - Get comparison charts data

## Notes

- The Flask app uses the same simulation backend as the Streamlit version
- Session data is stored in Flask's default session (cookie-based)
- For production, consider using a more secure session storage (Redis, database, etc.)

