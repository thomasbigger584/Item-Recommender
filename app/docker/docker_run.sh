#!/usr/bin/env bash
docker run -d \
    -p 8000:8000 \
    -v ~/volumes/item-recommender/trained_models:/usr/src/app/app/trained_models \
    item-recommender:latest