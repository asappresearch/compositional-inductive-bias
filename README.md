# Icy code

To run inductive bias experiments, see below. Note that many scripts assume that `MLFLOW_TRAKCING_URI` exists, and points to a valid mlflow server.

## Examples of grammars:

Use `jupyter/mll/sample_grammars.ipynb`

## Evaluation of Icy benchmark grammars using compositional metrics

For metrics except tre:
```
python mll/sample_grammar_rhos.py --out-csv cmet005.csv --meanings 5x10 --metrics posdis,bosdis,compent,topsim
```

For TRE:
```
python mll/sample_grammar_tres.py --out-csv ibg011.csv
```

## Human evaluation

### Web site

The code for the web-site is in `mll/turk/turk-web`. You need to have `npm` installed. Then from this folder, run:
```
npm install
npm run build
```
- The `build` subdirectory will then contain the built website
- The website will be served from your webservice, see below

### Webservice

To run the webservice, you need to first build the web site (above), then, with python 3.7+ installed, first install the requirements`
```
pip install -r mll/turk/webservice/requirements.txt
```
Then, to run the webservice:
```
aihttp-devtools runserver mll/turk/webservice/service.py -p 5000
```
- this will run on port 5000 by default
- this will also serve the website, from the `mll/turk/web-site/build` folder, as the sub-folder `/web/`

### https/ssl

You will need to place an https/ssl reverse proxy in front of the webservice. Nginx was used for this, and an example configuration file is pasted into the head of service.py file.

You will need to configure a dns name to point to your web-hosting, in order to obtain an https/ssl certificate. We used LetsEncrypt to create our certificate.

### MTurk

From an Amazon Mechanical Turk 'Requester' account, create a 'survey' task, and edit the url to point to your published webservice
- you need to add the querystring parameter `?taskId=[task type]` to the url, where `[task_type]` is the name of one of the files from `mll/turk/webservice/configs` folder, without the `.yaml` extension
- you need to provide the url of the published `index.html` from the `build` subdirectory, which the webservice publishes as the `web` sub-folder
- the url will likely look something like `https://mydomain.com/web/index.html?taskId=eng_pnts_rot`
    - (where you will replace `mydomain.com` with the actual dns name that you are publishing the webservice under)

### Available configs

*`synth` dataset:*
- pnts_comp
- pnts_perm
- pnts_proj
- pnts_rot
- pnts_shufdet

*`eng` dataset:*
- eng_pnts_comp
- eng_pnts_perm
- eng_pnts_proj
- eng_pnts_rot
- eng_pnts_shufdet

### Database

- `service.py` will create an SQLite database at `data/turk.db`

### Accepting/rejecting HITs

- you can use the script `mll/turk/tools/review_hits.py` to review hits, and accept/reject them
    - by default, the script rejects all HITs which score 0, and accepts automatically all HITs with a score of 100 or more
    - other hits are presented to you, for review. you can hit `a` to accept, `n` to reject, `s` to skip

### Result analysis

- you can use the script `mll/turk/tools/reduce_results.py` to retrieve the results, from the database, and write into latex formatted tables.

## Evaluation of standard neural models

### Sender models

```
for seed in {123..127}; do { python mll/mem_runner2.py --ref foo${seed} --dir send --tgt-acc 0.8 -m 5x10 --count-steps --seed ${seed}; } done
```

You can synthesize the resulting logs using:

```
python mll/analyse/reduce_csvs.py --in-refs foo123-foo127 \
    --out-ref foo \
    --row-key arch \
    --include-fields arch,params,Permute_ratio,RandomProj_ratio,Cumrot_ratio,ShuffleWordsDet_ratio,Holistic_ratio 
    --numeric-fields Permute_ratio,RandomProj_ratio,Cumrot_ratio,ShuffleWordsDet_ratio,Holistic_ratio \
    --include-keys FC1L,FC2L,RNNAutoReg:LSTM,RNNAutoReg2L:LSTM,TransDecSoft,TransDecSoft2L,Hashtable
```

### Receiver models

```
for seed in {123..127}; do { python mll/mem_runner2.py --ref foo${seed} --dir recv --tgt-acc 0.8 -m 5x10 --count-steps --seed ${seed}; } done
```

You can again use `mll/analyse/reduce_csvs.py` to reduce the resulting logs files.

## Effect of number of parameters

```
for seed in {123..132}; do { python mll/mem_runner2.py --ref foo${seed} --dir send --tgt-acc 0.8 -m 5x10 --count-steps --send-arch RNNAutoReg:LSTM,FC2L --embedding-size 1280 --render-every-steps 1 --seed ${seed}; } done
```

Then reduce the results using `reduce_csvs.py`, as above

## Acquisition accuracy given fixed training budget

```
for seed in {123..132}; do { ulfs_submit.py --ref foo${seed} -- mll/mem_runner2.py --dir send --max-mins 60 --seed ${seed}; } done
```

Reduce the results using
```
python mll/analyse/reduce_csvs.py --in-refs foo123-foo132 --out-ref foo --row-key arch --include-fields arch,params,Permute,RandomProj,Cumrot,ShuffleWordsDet,Holistic --numeric-fields Permute,RandomProj,Cumrot,ShuffleWordsDet,Holistic --include-keys FC1L,FC2L,RNNAutoReg:LSTM,RNNAutoReg2L:LSTM,TransDecSoft,TransDecSoft2L,Hashtable
```

## Search for model with low bias against shufdet

```
for seed in {123..132}; do { python mll/mem_runner2.py --ref foo${seed} --dir send --tgt-acc 0.8 --send-arch RNNAutoReg:LSTM,RNNZero:LSTM,HierAutoReg:RNN,HierZero:RNN,RNNAutoReg:dgsend,RNNZero:dgsend,HierAutoReg:dgsend,HierZero:dgsend --count-steps --seed ${seed}; } done
```

Then reduce the results using:

```
python mll/analyse/reduce_csvs.py --in-refs foo123-foo132 --out-ref foo --row-key arch --include-fields arch,params,Permute_ratio,RandomProj_ratio,Cumrot_ratio,ShuffleWordsDet_ratio,Holistic_ratio --numeric-fields Permute_ratio,RandomProj_ratio,Cumrot_ratio,ShuffleWordsDet_ratio,Holistic_ratio
```

## End to end training

```
for seed in {123..132}; do { python mll/mem_runner_e2e.py --ref foo${seed} --link RL --train-acc 0.99 --send-ent-reg 0.01 --recv-ent-reg 0.01 --opt Adam --softmax-sup --seed ${seed} --arch-pairs FC2L+RNN:LSTM,RNNAutoReg:LSTM+RNN:LSTM,TransDecSoft+RNN:LSTM,HierZero:dgsend+RNN:LSTM; } done
```

Then, to output csv of results:
```
python mll/analyse/graph_example_training_curves_2.py --in-ref foo123-foo132 --out-csv /tmp/foo.csv
```

Or to plot a png:
```
python mll/analyse/graph_example_training_curves_2.py --in-ref foo123-foo132 --out-png /tmp/out.png
```
