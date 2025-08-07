from flask import Flask
from flask_cors import CORS
# Correct relative import for the blueprint
from .routes.rag_routes import rag_routes

def create_app():
    app = Flask(__name__)
    CORS(app)

    # CRITICAL: Register the blueprint here
    app.register_blueprint(rag_routes)

    return app
