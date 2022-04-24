from http.server import BaseHTTPRequestHandler, HTTPServer
import time
import asyncio
import threading
from datetime import datetime
import sys


class ConsoleServer(HTTPServer):
    def __init__(self, 
                 server_address=("192.168.1.74", 8080), 
                 index_name="ConsoleIndex.html", 
                 resource_path="./UFE/utilities/resource/",
                 name="Python Console",
                 console_capacity=100):
        super().__init__(server_address, ConsoleServerHandler)
        self.thread = None
        self.resource_path = resource_path
        self.index_name = index_name
        self.index = None
        self.name = name
        self.console_msgs = []
        self.stdout = sys.stdout
        self.skip = False
        self.console_capacity = console_capacity
        
        self.load_resouces()

    def load_resouces(self):
        with open(self.resource_path + self.index_name, "r") as f:
            self.index = f.read()
    
    def start(self):
        if (self.thread is None):
            self.thread = threading.Thread( target=self.serve_forever)
            self.thread.start()
            sys.stdout = self
            print(f"Server started on {self.server_address[0]}:{self.server_address[1]}")
        else:
            print("Server is already running")

    def stop(self):
        self.shutdown()
        self.thread.join()
        self.thread = None
        sys.stdout = self.stdout

    def write(self, text): 
        self.skip = not self.skip
        if (self.skip):
            self.console_msgs.append(text)
            if (len(self.console_msgs) > self.console_capacity):
                self.console_msgs.pop(0)
        sys.stdout = self.stdout
        print(text, end="")
        sys.stdout = self

class ConsoleServerHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(bytes(self.format_index(), "utf-8"))

    def format_index(self):
        now = datetime.now()
        rules = {
            "%console_name%" : self.server.name,
            "%date%" : now.strftime("%Y-%m-%d"),
            "%time%" : now.strftime("%H:%M:%S"),
            "%console_output%" : "No Output"
            }
        html_output = []
        for index, msg in enumerate(self.server.console_msgs):
            if (index % 2 == 0):
                html_output.append(f"<tr class='even'><td>{msg}</td></tr>")
            else:
                html_output.append(f"<tr class='odd'><td>{msg}</td></tr>")
        rules["%console_output%"] = "".join(html_output)

        formatted_index = self.server.index
        for key, value in rules.items():
            formatted_index = formatted_index.replace(key, value)
        return formatted_index

