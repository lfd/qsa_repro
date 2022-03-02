# Reproduction package

Code and artifacts accompanying the paper 

```
@inproceedings{Schoenberger.2021,
  author    = {Schönberger, Manuel and Franz, Maja and Scherzinger, Stefanie and Mauerer, Wolfgang},
  title     = {Peel | Pile? {Cross}-Framework Portability of Quantum Software},
  booktitle = {19th IEEE International Conference on Software Architecture (ICSA)},
  address   = {Honolulu, HI, USA},
  publisher = {IEEE},
  year      = {2022}
}
```

## Docker 

### Get docker image
Build image: 

```docker build -t qsa-repro .```

or pull image: 

```docker pull ghcr.io/lfd/qsa-repro/qsa-repro:latest```

### Create Container

```docker run --name qsa-repro -it qsa-repro [<-flags>] [<option>]```

The `<option>` specifies which operations are performed on container start.

Available options are:
* `experiments_only`: performs the operations of the experimental Analysis (RL trainings and MQOs)\*
* `rl_only`: performs RL trainings\*
* `mqo_only`: performs MQO experiments
* `paper_only`: generates the full paper from LaTeX
* `all`: performs all of the above\*
* `bash`(default): does not perform any operation, but launches interactive shell

Feel free to define additional `<-flags>`, e.g.:
* Volume, to keep track of generated files on the host system: `-v $PWD:/home/repro/qsa-repro`
* Port forwarding to launch TensorBoard on the container to track the training process for RL on the host: `-p 6006:6006`. TensorBoard can be started in the Container with: `tensorboard --logdir expAnalysis/RL/logs --host 0.0.0.0`

\*Please note the long runtimes for RL trainings (several days). For quickly inspecting our reproduction package, we recommend to use the options `mqo_only`, `paper_only` or `bash`.  
