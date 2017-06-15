#!/bin/bash
coverage run --omit=*/python2.7/*,*test_*py fffsat/tests/test_*py
coverage report -m
