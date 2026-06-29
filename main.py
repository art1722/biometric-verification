"""FastAPI entrypoint for the biometric QC service.

This is the `app` object uvicorn looks for. The Dockerfile starts it with:
    uvicorn main:app --host 0.0.0.0 --port 8000

Two ways to run it locally (outside Docker):
    uvicorn main:app --reload        # recommended for development (auto-reload)
    python main.py                   # uses the __main__ block at the bottom

The QC pipeline itself (qc/, run_face.py, ...) is untouched — this file only
exposes it over HTTP. Wire real endpoints (e.g. POST a video -> run_face_rgb)
on top of this skeleton later.
"""

from fastapi import FastAPI

app = FastAPI(title="Biometric QC API")


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/health")
async def health():
    # A tiny endpoint that just says "I'm alive". Containers and load balancers
    # ping this to know the service started correctly.
    return {"status": "ok"}


if __name__ == "__main__":
    # Lets you run `python main.py` directly. uvicorn.run imports "main:app"
    # the same way the CLI does. reload=True is convenient locally; the Docker
    # CMD does NOT use reload (you don't want auto-reload in a deployed image).
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)