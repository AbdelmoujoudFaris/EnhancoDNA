# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in EnhancoAI (e.g. unsafe
deserialization of structure/trajectory/checkpoint files, path traversal
in file loading, or a dependency vulnerability), please report it
privately by opening a GitHub security advisory on this repository rather
than a public issue.

Please include:

- A description of the vulnerability and its potential impact.
- Steps to reproduce (a minimal example file or command, if applicable).
- Any suggested mitigation.

## Scope notes

- EnhancoAI loads structure files (PDB/mmCIF), MD trajectories, and
  PyTorch checkpoints from user-supplied paths. Loading a checkpoint calls
  `torch.load`; only load checkpoints you trust, as with any
  pickle-based format.
- The GUI does not transmit data over the network. The CLI and scripts
  only read/write local files unless explicitly configured otherwise.

We aim to acknowledge reports within 5 business days.
