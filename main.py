from typing import Optional

from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware

import tensorflow_hub as hub

from google.cloud import storage

import tensorflow as tf
import tensorflow_text

app = FastAPI()

origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=methods,
    allow_headers=headers
)


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"

    # The ID of your GCS object
    # source_blob_name = "storage-object-name"

    # The path to which the file should be downloaded
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    # Construct a client side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
    # any content from Google Cloud Storage. As we don't need additional data,
    # using `Bucket.blob` is preferred here.
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print('Blob {} downloaded to {}.'.format(source_blob_name, destination_file_name))


def download_model_from_cloud():
    download_blob(
        bucket_name="classify_engine_demo_ml_models",
        source_blob_name="pets_friendly_property_query_classification.h5",
        destination_file_name="pets_friendly_property_query_classification.h5"
    )


def load_model():
    dataset_name = 'pets_friendly_property_query_classification'
    saved_model_path = dataset_name + ".h5"
    reloaded_model = tf.keras.models.load_model(saved_model_path, custom_objects={'KerasLayer': hub.KerasLayer})
    return reloaded_model


download_model_from_cloud()
model = load_model()


@app.get("/")
def read_root():
    return {"service_name": "Pet-friendly property search query detection", "author": "Classify Engine"}


@app.get("/predict")
def predict(text: Optional[str] = None):
    reloaded_results = [float(pred) for pred in list(model(tf.constant([text])).numpy()[0])]
    labels = ["unrelated", "pet", "dog", "cat"]
    print(reloaded_results)
    return {
        'text': text,
        "predictions": reloaded_results,
        "labels": labels,
        "predicted_class": labels[reloaded_results.index(max(reloaded_results))]
    }
