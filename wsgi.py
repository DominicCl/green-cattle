import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from app.server import app

if __name__ == "__main__":
    app.run(port=8080)