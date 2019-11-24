#!/usr/bin/env bash
# find ./config_bash/ -name "*.sh" -delete -print
# find ./config_yaml/ -name "*.yaml" -delete -print
find ./jobs/ -name "*.sl" -delete -print
find ./outputjobs/ -name "*.e*" -delete -print
find ./outputjobs/ -name "*.o*" -delete -print
# cd exps/search_optimizer/
# rm -rv adadelta
# cd ../..
