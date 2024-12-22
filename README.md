# ψDAG: Projected Stochastic Approximation Iteration for DAG Structure Learning

This repository contains the implementation of [**ψDAG**](https://arxiv.org/abs/2410.23862), a novel framework for learning Directed Acyclic Graphs (DAGs) using a Stochastic Approximation approach integrated with Stochastic Gradient Descent (SGD)-based optimization techniques. ψDAG introduces new projection methods tailored to efficiently enforce DAG constraints, ensuring convergence to a feasible local minimum with improved computational efficiency.

---

## Overview

Learning the structure of DAGs is a fundamental challenge in machine learning due to the vast combinatorial search space, which scales exponentially with the number of nodes. Traditional methods often face optimization difficulties due to the highly non-convex nature of DAG constraints and high per-iteration computational complexity. 

ψDAG addresses these challenges by:
- Employing a Stochastic Approximation approach combined with SGD-based optimization.
- Introducing projection methods that efficiently enforce DAG constraints.
- Achieving low iteration complexity, making it suitable for large-scale problems.

Comprehensive experimental evaluations demonstrate the effectiveness and scalability of ψDAG across various settings.

---

## Installation

To use ψDAG, clone this repository and install the required dependencies:

```bash
git clone https://github.com/kz29/-psiDAG.git
cd psidag
pip install -r requirements.txt
``` 
---
## BibTeX
If you find this work useful, please cite our paper:
```
@article{ziu2024psi,
  title={$$\backslash$psi $ DAG: Projected Stochastic Approximation Iteration for DAG Structure Learning},
  author={Ziu, Klea and Hanzely, Slavom{\'\i}r and Li, Loka and Zhang, Kun and Tak{\'a}{\v{c}}, Martin and Kamzolov, Dmitry},
  journal={arXiv preprint arXiv:2410.23862},
  year={2024}
}
```
