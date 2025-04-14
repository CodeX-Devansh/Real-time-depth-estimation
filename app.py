import asyncio
import uuid
import shutil
import os
import time
import base64
import io
import traceback
import json # Import json for decoding potential errors

import torch
import numpy as np
import cv2
# NOTE: Matplotlib is no longer needed for real-time processing with OpenCV viz
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
from PIL import Image # Use Pillow for easier handling

from fastapi import FastAPI, Request, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError # Import specific exceptions

# --- Configuration ---
MODEL_TYPE = "MiDaS_small" # Using small for better real-time performance

# --- Model Loading ---
print("Loading MiDaS model...")
start_load_time = time.time()
midas_model = None
transform = None
device = None

try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Using trust_repo=True might be necessary depending on torch/hub versions
    midas_model = torch.hub.load("intel-isl/MiDaS", MODEL_TYPE, trust_repo=True)
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)

    if MODEL_TYPE == "MiDaS_small":
        transform = midas_transforms.small_transform
    else: # DPT_Hybrid or DPT_Large
        transform = midas_transforms.dpt_transform

    midas_model.to(device)
    midas_model.eval()
    load_time = time.time() - start_load_time
    print(f"MiDaS model '{MODEL_TYPE}' loaded successfully in {load_time:.2f} seconds.")

    # --- ADDED GPU USAGE VERIFICATION ---
    if midas_model:
        try:
            # Check the device of the first parameter (usually indicates where the model lives)
            model_device = next(midas_model.parameters()).device
            print(f"Verified: Model parameters are on device: {model_device}")
            if device != model_device:
                 print(f"Warning: Model device ({model_device}) doesn't match target device ({device}). Check model loading.")
        except Exception as e:
            print(f"Could not verify model device: {e}")
    # --- END GPU USAGE VERIFICATION ---

except Exception as e:
    print(f"FATAL ERROR: Could not load MiDaS model '{MODEL_TYPE}' from torch.hub.")
    print(f"Error details: {e}")
    traceback.print_exc()
    print("Please ensure internet connectivity, correct dependencies, and MODEL_TYPE.")
    import sys
    sys.exit(1)

# --- Helper Function: Real-time Depth Prediction (OpenCV Optimized) ---
def predict_depth_realtime(img_rgb: np.ndarray, colormap: str = 'plasma') -> str:
    """
    Optimized version using OpenCV for colormapping.
    Takes NumPy RGB array, returns base64 encoded JPEG data URL of the depth map visualization.
    """
    if img_rgb is None or img_rgb.size == 0:
        raise ValueError("Input image is empty")
    if transform is None or midas_model is None or device is None:
        raise RuntimeError("Model or transform not initialized")

    start_pred_time = time.time()
    try:
        # --- Core MiDaS prediction logic (same) ---
        input_batch = transform(img_rgb).to(device)

        with torch.no_grad():
            prediction = midas_model(input_batch)
            # Interpolate prediction to match input size
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic", # Can try 'bilinear' for small model if bicubic is slow
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()

        # --- Visualization (Optimized with OpenCV) ---
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        if depth_max - depth_min > 1e-6:
            # Normalize depth map to 0-1 range
            normalized_map = (depth_map - depth_min) / (depth_max - depth_min)
        else:
            # Handle case where depth map is flat
            normalized_map = np.zeros_like(depth_map)

        # Convert normalized map to 8-bit unsigned integer (0-255 scale)
        depth_8bit = (normalized_map * 255).astype(np.uint8)

        # Map colormap string name to OpenCV colormap constant
        cv2_colormap_map = {
            "viridis": cv2.COLORMAP_VIRIDIS,
            "plasma": cv2.COLORMAP_PLASMA,
            "magma": cv2.COLORMAP_MAGMA,
            "inferno": cv2.COLORMAP_INFERNO,
            "cividis": cv2.COLORMAP_CIVIDIS,
            "gray": -1, # Special flag for grayscale
            "jet": cv2.COLORMAP_JET,
            "hot": cv2.COLORMAP_HOT,
            "cool": cv2.COLORMAP_COOL,
            # Add more OpenCV colormaps if desired
        }
        # Get the corresponding OpenCV code, default to plasma if not found
        cv2_colormap = cv2_colormap_map.get(colormap.lower(), cv2.COLORMAP_PLASMA)

        if cv2_colormap == -1: # Handle grayscale case
            # Convert grayscale to 3-channel BGR for consistent processing downstream
            depth_viz = cv2.cvtColor(depth_8bit, cv2.COLOR_GRAY2BGR)
        else:
            # Apply the selected OpenCV colormap
            depth_viz = cv2.applyColorMap(depth_8bit, cv2_colormap)

        # --- Encoding (Optimized to JPEG) ---
        # Set JPEG encoding quality (adjust 0-100, higher quality = larger size)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        # Encode the BGR visualization image to JPEG format in memory
        is_success, buffer = cv2.imencode(".jpg", depth_viz, encode_param)

        if not is_success:
            raise RuntimeError("Failed to encode depth map to JPEG")

        # Encode the JPEG byte buffer to base64 string
        encoded_result = base64.b64encode(buffer).decode("utf-8")
        # Format as a base64 data URL for direct use in HTML/JS
        result_data_url = f"data:image/jpeg;base64,{encoded_result}" # Correct mime type

        pred_time = time.time() - start_pred_time
        # Uncomment for performance debugging:
        # print(f"Frame processed (OpenCV viz) in {pred_time:.3f} seconds")
        return result_data_url

    except Exception as e:
        # No Matplotlib figures to close in this version
        print(f"Error during optimized depth prediction function: {e}")
        traceback.print_exc()
        raise # Re-raise the exception to be handled by the caller

# --- FastAPI Application Setup ---
app = FastAPI(title="MiDaS Real-time Depth Estimation")

# --- WebSocket Endpoint ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print(f"WebSocket client connected: {websocket.client}")
    close_code = None # Variable to store the close code if disconnected
    client_disconnected_cleanly = False
    try:
        while True:
            # Receive text data (expecting JSON string)
            raw_data = await websocket.receive_text() # Use receive_text for robustness

            # Attempt to parse the received text as JSON
            try:
                 payload_json = json.loads(raw_data)
            except json.JSONDecodeError:
                 print(f"Warning: Received non-JSON message: {raw_data[:100]}...") # Log snippet
                 continue # Ignore non-JSON messages

            # Extract data from the parsed JSON
            frame_data_url = payload_json.get("frameDataUrl")
            colormap = payload_json.get("colormap", "plasma") # Default colormap if not provided

            if not frame_data_url:
                print("Warning: Received JSON payload without 'frameDataUrl'.")
                continue

            # Decode base64 data URL (sent as JPEG from frontend)
            try:
                header, encoded = frame_data_url.split(",", 1)
                image_data = base64.b64decode(encoded)
                # Use Pillow to open image from bytes, ensure it's RGB
                img_pil = Image.open(io.BytesIO(image_data)).convert("RGB")
                # Convert PIL image to NumPy array (OpenCV format)
                img_rgb = np.array(img_pil)
            except Exception as decode_err:
                 print(f"Error decoding frame: {decode_err}")
                 traceback.print_exc()
                 continue # Skip this frame if decoding fails

            # --- Run Depth Prediction ---
            try:
                # Call the optimized prediction function
                result_data_url = predict_depth_realtime(img_rgb, colormap=colormap)

                # Send result back - Wrap in TRY/EXCEPT for closed connections
                try:
                     # Check connection state *before* attempting to send
                     if websocket.client_state == WebSocketState.CONNECTED:
                          await websocket.send_text(result_data_url)
                     else:
                          # If state is already disconnected before send
                          print("Client disconnected before send attempt.")
                          client_disconnected_cleanly = True # Assume clean if state changed
                          break # Exit the main processing loop

                # Catch specific WebSocket closure exceptions during send
                except (ConnectionClosedOK, ConnectionClosedError) as send_close_err:
                     print(f"Connection closed during send: {type(send_close_err).__name__}")
                     client_disconnected_cleanly = isinstance(send_close_err, ConnectionClosedOK)
                     close_code = send_close_err.code
                     break # Exit the main processing loop
                except Exception as send_err:
                     # Catch other potential errors during the send operation
                     print(f"Error sending data: {send_err}")
                     traceback.print_exc()
                     # Usually indicates a connection issue, safest to break
                     break # Exit the main processing loop

            except Exception as pred_err:
                 # Errors from predict_depth_realtime are logged within that function
                 print(f"Error during prediction processing loop (after prediction call): {pred_err}")
                 # Decide whether to break or continue on prediction errors
                 # Let's continue for now to potentially process the next frame
                 continue

    # --- Handle Disconnections Detected During Receive ---
    except WebSocketDisconnect as e:
        close_code = e.code # Store the close code from the exception
        print(f"WebSocket client initiated disconnect (Code: {close_code})")
        client_disconnected_cleanly = True # WebSocketDisconnect implies a cleaner close
    except (ConnectionClosedOK, ConnectionClosedError) as close_err:
         # Catch disconnects detected by receive_text()
         close_code = close_err.code
         print(f"Connection closed during receive (Code: {close_code}): {type(close_err).__name__}")
         client_disconnected_cleanly = isinstance(close_err, ConnectionClosedOK)
    except Exception as e:
         # Catch other potential errors during receive (e.g., message format issues not caught above)
         print(f"Unhandled WebSocket Error during receive phase: {e}")
         traceback.print_exc()

    # --- Cleanup Phase ---
    finally:
         print("WebSocket connection closing sequence starting.")
         # Print the captured close code if available
         if close_code is not None:
             print(f"Disconnect Code was: {close_code}")
         else:
             print("Disconnect Code not captured (may indicate abnormal or unhandled closure).")

         # Check connection state before attempting to close explicitly from server-side
         # Only try to close if it wasn't a clean disconnect initiated by client/server already
         if websocket.client_state == WebSocketState.CONNECTED and not client_disconnected_cleanly:
               print("Attempting to close server-side connection due to error or unexpected state.")
               try:
                    # Send a standard "Internal Server Error" close code
                    await websocket.close(code=1011)
               except Exception as close_final_err:
                    # Log errors during the final close attempt
                    print(f"Error during final websocket close attempt: {close_final_err}")
         elif websocket.client_state == WebSocketState.DISCONNECTED:
              print("Connection already marked as disconnected.")
         else:
              # Log any other state found during cleanup
              print(f"WebSocket final state before exit: {websocket.client_state}")

         print(f"WebSocket endpoint finished for client: {websocket.client}")


# --- Serve Frontend ---
@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    """Serves the main HTML frontend."""
    html_file_path = "index.html"
    if not os.path.exists(html_file_path):
        raise HTTPException(status_code=404, detail="index.html not found")
    try:
        with open(html_file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content)
    except Exception as e:
        print(f"Error reading index.html: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error reading frontend file.")


# --- Main execution block (for local testing) ---
if __name__ == "__main__":
    import uvicorn

    print("Starting FastAPI server for LOCAL DEVELOPMENT...")
    port = int(os.environ.get("PORT", 8000))
    # Change host to "0.0.0.0" to allow access from other devices on your network
    # Use "127.0.0.1" to restrict access to only your local machine
    host = "127.0.0.1"

    uvicorn.run(
        "app:app", # Point to the FastAPI app instance in this file (app.py)
        host=host,
        port=port,
        reload=True, # Automatically restart server on code changes
        log_level="info", # Set desired logging level
        ws_ping_interval=25, # Send keep-alive pings every 25 seconds
        ws_ping_timeout=20,  # Wait 20 seconds for pong response
        timeout_keep_alive=30, # Keep underlying TCP connection alive longer
    )
