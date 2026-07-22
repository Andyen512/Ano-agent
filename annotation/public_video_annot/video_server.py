#!/usr/bin/env python3
"""Simple HTTP server for serving video files directly."""
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
from functools import partial

VIDEO_ROOT = "/data_4/liuyuan/lifebench/data/public_data"

class CORSRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=VIDEO_ROOT, **kwargs)

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'public, max-age=86400')
        super().end_headers()

if __name__ == "__main__":
    port = 5004
    server = HTTPServer(("0.0.0.0", port), CORSRequestHandler)
    print(f"Video server running on http://0.0.0.0:{port}")
    server.serve_forever()
