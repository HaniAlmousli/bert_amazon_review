#!/bin/sh

image=$1

mkdir -p test_dir/model
mkdir -p test_dir/output

rm -r test_dir/model/*
rm -r test_dir/output/*

docker run --gpus all -v $(pwd)/test_dir:/opt/ml --rm ${image} train
