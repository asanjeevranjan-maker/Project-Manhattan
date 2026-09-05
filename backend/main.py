from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(
    title=" SatQuery AI Backend\,
 description=\SatQuery Grounding DINO and Multi-Temporal Satellite Analysis API\,
 version=\2.0.0\
)

app.add_middleware(
 CORSMiddleware,
 allow_origins=[\*\],
 allow_credentials=True,
 allow_methods=[\*\],
 allow_headers=[\*\],
)

@app.get(\/\)
def read_root():
 return {
 \status\: \online\,
 \service\: \SatQuery AI API\,
 \docs\: \/docs\
 }

@app.get(\/health\)
def health_check():
 return {
 \status\: \healthy\,
 \service\: \SatQuery AI API\
 }

if __name__ == \__main__\:
 import uvicorn
 uvicorn.run(app, host=\127.0.0.1\, port=8000)
