#!/usr/bin/env bash
docker run \
    -p 8000:8000 \
    -v ~/volumes/item-recommender/trained_models:/usr/src/app/app/trained_models \
    item-recommender:latest