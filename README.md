# compositional-inductive-bias
Code for work "A Framework for Measuring Compositional Inductive Bias", by Perkins, 2021

## To run

### Sender models

e.g.
```
python mll/mem_runner2.py --ref myexpref123 --dir send --max-mins 120 --tgt-acc 0.95 --rl --send-ent-reg 0.1 --seed 123 --opt Adam
```

### Receiver models

e.g.
```
python mll/mem_runner2.py --ref myexpref123 --dir recv --max-mins 120 --rl --tgt-acc 0.95 --ent-reg 0.1 --seed 125 --opt Adam
```

### End to end models

e.g.
```
python mll/mem_runner_e2e.py --ref myexpref123 --link RL --train-acc 0.99 --send-ent-reg 0.01 --recv-ent-reg 0.01 --opt Adam --softmax-sup --seed 125 --max-e2e-steps 10000
```

### End to end from scratch

Option 1: run one at a time

e.g.
```
python mll/e2e_fixpoint.py --ref myexpref123 --meanings 3x10 --train-acc 0.0 --link RL --softmax-sup --send-arch HierZero:dgsend --recv-arch Hier:dgrecv --max-e2e-steps 55000
```

Option 2: run in bulk

Pre-requisite: the following scripts must exist, and be in the PATH:
```
ulfs_logs.sh [ref]
  - dumps the logs from job with name [ref]

ulfs_submit.py -r [ref] --no-follow -- [script_path] [script args]
  - runs script [script_path], passing in args [script args]
  - [ref] is a job name, that will be alphanumeric plus hyphens
    also allowed
  - ulfs_submit.py, with the argument --no-follow, should simply
    start the job running, executing the script [script_path]
    asynchronously, and leaving it to run
```

Then to run, e.g.
```
python mll/submit_e2e_from_scratch.py --ref myexpref123 --meanings 2x33 --seed 123 --max-e2e-steps 55000
```

## Logging

If you export `MLFLOW_TRACKING_URI`, with a valid uri that points to an mlflow server, then many of these scripts will log to mlflow. In any case, they will log to a folder `logs`. Format of the `logs` logs is: text files, one record per line, in json format.

## To analyse results

Whilst it's possible to analyze the results by hand, the following scripts can be used to collate results.

### Sender results

e.g.
```
python mll/analyse/reduce_send_recv_results.py --refs-by-repr soft[expref1,expref2,expref3],gumb[expref4,expref5,expref6],discr[expref7,expref8,expref9] --out-ref foo --skip-max --no-stderr --dir send
```
... where `exprefnnn` are valid experiment references from the parameter `--ref` of the relevant runs. Results will be collated into representations `soft`, `gumb` and `discr`. The runs within each representation will be averaged, and the standard error of the mean is available. Results are provided in both .tex format, and in .csv format. These scripts assume that one is using a Mac, and run `open` on the output, so you might need to comment those lines out.

### Receiver results

As for sender results, but specify `--dir recv`

### End to end results

e.g.
```
python mll/analyse/reduce_e2e_sup_results.py --refs-by-repr soft[ibe107,ibe109,ibe114],gumb[ibe103,ibe110_copy,ibe111_copy],discr[ibe105,ibe112,ibe113] --direction send
```
Format of the `--refs-by-repr` parameter is as for sender results. Results are output in similar format as for sender results.

### End to end from scratch

This script requires that the end to end from scratch experiments were logged to mlflow, that the mlflow server is up and available, and that MLFLOW_TRACKING_URI contains a valid uri that points to the mlflow server.

Then run e.g.
```
python mll/analyse/reduce_e2efs_results.py --refs ibfs005,ibfs008,ibfs011 --k-steps 50
```
where `--refs` contains valid experiment refs from the `--ref` parameter of the runs one wants to collate. Output is again available in both .tex and .csv format.
