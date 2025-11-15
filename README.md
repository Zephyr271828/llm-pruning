# LLM Pruning Collection

The repo is organized as follows:

```
├── jobman
├── pruning
│   ├── LLM-Pruner
│   ├── llmshearing
│   ├── minitron
│   ├── shortened-llm
│   ├── shortgpt
│   └── wanda
└── training
    ├── fms
    ├── fms_fsdp
    └── maxtext
```
where `pruning` is the collection of the pruning methods we experimented; `training` contains the LLM training frameworks we used, and we provided options for both TPU and GPU; `jobman` is a TPU orchestration we developed to mimic the slurm system.

For an overview of the pruning methods, see [here](pruning/README.md); for usage of the training frameworks, see [here](training/README.md); for usage of JobMan, see [here](jobman/README.md).

## Roadmap
- [ ] complete pruning code cleaning. [details](pruning/README.md#roadmap)
- [x] complete training code cleaning. [details](training/README.md#roadmap)
- [ ] accelerate lm-eval-harness for maxtext.
- [ ] simplify the design of jobman