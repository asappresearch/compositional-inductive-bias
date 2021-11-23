#!/bin/bash
# will pull one command at a time from _in, move it to _processing, then
# run it
# finally move it to _done
# output will go in _out, with similar name as the input file

while true; do {
    echo .
    filename=$(ls _in | head -n 1)
    if [[ x$filename != x ]]; then {
        echo processing
        mv _in/$filename _processing/
        stdbuf -o 0 -e 0 bash _processing/$filename | tee -a _out/$filename.out 2>&1
        mv _processing/$filename _done
        echo done
    } fi
    sleep 1;
} done
