"""
Advanced Production-Ready RAG System
- Asynchronous Flask implementation
- ChromaDB for vector storage
- Azure OpenAI integration
- Multi-user support with conversation history
- Multiple file format support (.txt, .pdf, .docx)
"""

# Correct import path to reference 'app' inside the 'api' directory
from api.app import create_app
import os

# Create the Flask application instance
app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('FLASK_ENV', 'development') == 'development'
    app.run(debug=debug, host='0.0.0.0', port=port)
