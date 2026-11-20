# Architecture

The project separates dataset handling, preprocessing, model inference, and API
delivery. The FastAPI backend calls the reusable `firesight` inference service.
The browser frontend communicates with the backend over HTTP.

