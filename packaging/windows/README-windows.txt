PCD Server for Windows (x64)
============================

Requirements: 64-bit Windows 10 or 11 on a CPU with AVX2 (practically every
PC from 2015 onward). No other software is needed. The server runs on the CPU.

1. Download the model (about 1 GB, one time only)

   Double-click scripts\download-default-model.cmd
   It saves Qwen3.5-0.8B-Q8_0.gguf into the models folder and verifies it.

2. Start the server

   Double-click start-server.cmd
   Wait for:  pcd_server v1 listening on http://127.0.0.1:8090

   If Windows shows "Windows protected your PC", click "More info" and then
   "Run anyway". The program is not code-signed; this is expected.

3. Open http://127.0.0.1:8090/ in your browser.

   Interactive API docs: http://127.0.0.1:8090/docs

Press Ctrl+C in the server window to stop it.

Options (run from a Command Prompt in this folder):

   pcd_server.exe --help
   pcd_server.exe --port 9000

The full documentation is at https://github.com/stephanj/pcdServer
