#!/bin/bash
mycount=0; while (( $mycount < 10 )); do  python temp.py --test ijn;((mycount=$mycount+1)); done;