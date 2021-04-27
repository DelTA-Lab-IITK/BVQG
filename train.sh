#th main.lua -data ../../data/exp21 -nClasses 7 -cache ../../data/cache22 -epochSize 100 -netType alexnetowt
CUDA_VISIBLE_DEVICES=9 th train.lua -gpuid 1 |tee log.txt


##

