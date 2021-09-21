# YummyStew

A python library for Quality-Diversity optimization.

Note: This library is still in its initial development, expect aggressive changes in API.

## Overview

*YummyStew, tasty and flavorful.*

[Quality diversity (QD) optimization](https://arxiv.org/abs/2012.04322) is a subfield of
optimization where solutions are generated to simultaneously meet two criterias:

- **Quality**: Maximizing or minimizing on objective(s) to satisfy our optimization goal.
- **Diversity**: Covering the **feature space** with various solutions, where **feature**
  characterizes the way to resolve the problem.

A user of YummyStew selects three components to satisfy their demands in application:

- **Container** saves the best solutions generated in behavior spaces.
- **Operator** defines how the candidate solutions are generated and select them by
  **quality** and/or **diversity** criterias.
- **Algorithm** packs the **Container** and **Operator** together. It provides interfaces
  for asking new candidate solutions and telling the algorithm how candidates performed.
