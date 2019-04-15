#!/usr/bin/env bash
docker run -p 8000:8000 \ 
    -v ~/volumes/item-recommender/trained_models:/usr/src/app/app/trained_models \
    -v ~/volumes/item-recommender/data:/usr/src/app/app/data \
    docker.cloudspace.pw/item-recommender:latest