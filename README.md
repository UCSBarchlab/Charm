Charm 
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/UCSBarchlab/Charm.git/master)
=====

Charm is an interpreted DSL and runtime for writing/managing
closed-form high-level architecture models.

Charm was presented at ISCA'18 @ LA, CA.

### Prerequisites

Python (v3.6)

pyparsing (v2.2.0)

numpy (v1.12.1)

scipy (v0.18.1)

mcerp (v0.11)

sympy (v1.1.1)

lmfit (v0.9.9)

networkx (v2.1)

For SMT capabilities:

z3 (v4.6.0 with python binding)

### Example Usage

**Just click the ![Binder](https://mybinder.org/badge.svg) button above or visit https://mybinder.org/v2/gh/UCSBarchlab/Charm.git/master **

To start it from command line, first install by
```python setup.py install```
then run
```python -m Charm.interpreter.parser {source_file} {options}```

### Contact & Citation:

For general questions feel free to reach out to [Archlab @ UCSB](https://www.arch.cs.ucsb.edu/).

For immediate help with Charm, contact Weilong (cuiwl@cs.ucsb.edu).

To cite our work:

```
Weilong Cui, Yongshan Ding, Deeksha Dangwal, Adam Holmes, Joseph McMahan, Ali JavadiAbhari, Georgios Tzimpragos, Frederic T. Chong and Timothy Sherwood. "Charm: A Language for Closed-form High-level Architecture Modeling" in Proceedings of the International Symposium of Computer Architecture (ISCA) June 2018. Los Angeles, CA.
```
