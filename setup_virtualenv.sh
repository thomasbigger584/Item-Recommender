#!/usr/bin/env bash

#installing python3.6 on mac
#brew install python3
#brew unlink python 
#brew install --ignore-dependencies https://raw.githubusercontent.com/Homebrew/homebrew-core/f2a764ef944b1080be64bd88dca9a1d80130c558/Formula/python.rb --ignore-dependencies
#brew link --overwrite python

virtualenv -p python3.6 env
source env/bin/activate