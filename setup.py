import http.server
import socketserver
import os
import sys

# Define default port and directory
DEFAULT_PORT = 8000
DEFAULT_DIRECTORY = "."

class Handler(http.server.SimpleHTTPRequestHandler):
    """
    A simple HTTP request handler that serves files from the current directory.
    """
    def __init__(self, *args, **kwargs):
        # Set the directory for the handler
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def do_GET(self):
        """
        Overrides the do_GET method to handle requests.
        """
        # A simple print statement to show the request is being handled
        print(f"Serving file: {self.path}")
        return super().do_GET()

def run_server(port, directory):
    """
    Starts the HTTP server.
    """
    # Change the current working directory to the directory where the script is located
    # to ensure the server serves files relative to this script.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        with socketserver.TCPServer(("", port), Handler) as httpd:
            print(f"Serving files from directory '{directory}'")
            print(f"Server started at http://localhost:{port}")
            print("Press Ctrl+C to stop the server.")
            httpd.serve_forever()
    except OSError as e:
        print(f"Error: Could not start the server on port {port}. The port might be in use.", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
    except KeyboardInterrupt:
        print("\nServer stopped.")

if __name__ == "__main__":
    PORT = DEFAULT_PORT
    DIRECTORY = DEFAULT_DIRECTORY
    # A simple way to handle command-line arguments for port
    if len(sys.argv) > 1:
        try:
            PORT = int(sys.argv[1])
        except ValueError:
            print("Invalid port number provided. Using default port 8000.", file=sys.stderr)
    
    run_server(PORT, DIRECTORY)