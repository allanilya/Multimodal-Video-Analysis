"""
Main application entry point
"""
import logging
from app import create_app, socketio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create app
app = create_app()

if __name__ == '__main__':
    # Run with SocketIO on port 5001 (5000 is used by macOS Control Center)
    socketio.run(
        app,
        host='0.0.0.0',
        port=5001,
        debug=True,
        allow_unsafe_werkzeug=True
    )
