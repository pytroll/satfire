#!/bin/bash
coverage run --omit=*/python2.7/*,*test_*py,*init*py,*version.py,setup.py,*/satpy/*,*/trollimage/* setup.py test
coverage report -m
