# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import uvicorn
from http import HTTPStatus
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from pydantic import BaseModel
from typing import Optional, List, Annotated
from .config import config
from .document import validate_files, save_files_to_tmp
from .evaluate import Evaluator
from .logger import logger


PUBLIC_API_PATH = os.getenv("PUBLIC_API_PATH", "/api").rstrip("/")

app = FastAPI(
    title=config.APP_DISPLAY_NAME,
    root_path="/v1/eval",
    root_path_in_servers=False,
    servers=[{"url": PUBLIC_API_PATH}],
    docs_url=None,
    redoc_url=None,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),  # Adjust this to your needs
    allow_credentials=True,
    allow_methods=os.getenv("CORS_ALLOW_METHODS", "*").split(","),
    allow_headers=os.getenv("CORS_ALLOW_HEADERS", "*").split(","),
)

evaluator = Evaluator(
    bert_scorer_model_name=config.BERT_SCORER_MODEL_ID,
    sbert_model_name=config.SBERT_MODEL_ID,
    nli_model_name=config.NLI_MODEL_ID,
    nli_model_revision=config.NLI_MODEL_REVISION
)


@app.get("/docs", include_in_schema=False)
async def swagger_ui():
    return get_swagger_ui_html(
        openapi_url=f"{PUBLIC_API_PATH}{app.openapi_url}",
        title=f"{app.title} - Swagger UI",
    )

class EvaluateData(BaseModel):
    generated: str
    reference: str
    question: str = ""
    metrics: Optional[List[str]] = None

@app.get(
    "/health",
    tags=["Status APIs"],
    summary="Check the health of the API service"
)
async def check_health():
    """
    Checks the health status of the application.
    This asynchronous function is used to verify that the application is running
    and healthy by returning a simple status message.

    Returns:
        dict: A dictionary containing the health status of the application.
    """

    return {"status": "Success", "message": "Service is up and running."}


@app.post(
    "/semantic-score",
    tags=["Evaluation APIs"],
    summary="Get semantic score from the datasets",
)
def get_semantic_score(input_data: EvaluateData):
    try:
        return evaluator._calculate_semantic_score(input_data.generated, input_data.reference)

    except Exception as e:
        logger.exception("semantic-score evaluation failed")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="Evaluation failed. Contact support if this persists."
        )


@app.post(
    "/bert-score",
    tags=["Evaluation APIs"],
    summary="Get BERT score from the datasets",
)
def get_bert_score(input_data: EvaluateData):
    try:
        return evaluator._calculate_bert_score(input_data.generated, input_data.reference)

    except Exception as e:
        logger.exception("bert-score evaluation failed")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="Evaluation failed. Contact support if this persists."
        )


@app.post(
    "/rouge-score",
    tags=["Evaluation APIs"],
    summary="Get ROUGE score from the datasets",
)
def get_rouge_score(input_data: EvaluateData):
    try:
        return evaluator._calculate_rouge_score(input_data.generated, input_data.reference)

    except Exception as e:
        logger.exception("rouge-score evaluation failed")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="Evaluation failed. Contact support if this persists."
        )


@app.post(
    "/average-score",
    tags=["Evaluation APIs"],
    summary="Get average score from the datasets for different metrics",
)
def get_average_score(input_data: EvaluateData):
    try:
        return evaluator._calculate_average_scores([(input_data.generated, input_data.reference)])

    except Exception as e:
        logger.exception("average-score evaluation failed")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="Evaluation failed. Contact support if this persists."
        )


@app.post(
    "/factual-entailment",
    tags=["Evaluation APIs"],
    summary="Get factual entailment label from the datasets",
)
def get_factual_entailment(input_data: EvaluateData):
    try:
        return evaluator._evaluate_factual_consistency(input_data.generated, input_data.reference)

    except Exception as e:
        logger.exception("factual-entailment evaluation failed")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="Evaluation failed. Contact support if this persists."
        )


@app.post(
    "/evaluate",
    tags=["Evaluation APIs"],
    summary="Evaluate datasets for accuracy",
)
async def evaluate_video_accuracy(
    file: Annotated[
        UploadFile,
        File(description="Upload one file containing generated and reference data.")
    ],
):
    try:
        status = validate_files([file])
        if status is False:
            logger.exception("Unsupported file format.")
            raise HTTPException(
                status_code=HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                detail="Unsupported file format. Please upload a .md or .tsv file."
            )

        # Save the file in /tmp/documents to load it later
        tmp_files = await save_files_to_tmp([file])
        if tmp_files is None or len(tmp_files) == 0:
            logger.exception(f"Error saving file.")
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                detail="Error saving file."
            )

        result = evaluator.run_evaluation_from_file(tmp_files[0])

        return result

    except HTTPException:
        # Re-raise HTTPException without modification
        raise

    except Exception as e:
        logger.exception("evaluate request failed")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="Evaluation failed. Contact support if this persists."
        )


if __name__ == "__main__":
    # Only this direct-run path defaults to loopback; the Dockerfile's uvicorn CLI
    # invocation controls the container's bind address independently via --host.
    uvicorn.run("app", host=os.getenv("UVICORN_HOST", "127.0.0.1"), port=9000)
