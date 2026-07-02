# Temporal-graph persistence estimation (partial observation)

You will estimate a property of large temporal graphs you can only observe partially.

**Definition.** Each graph's timeline is split into **5 equal time windows**. An edge (a pair of nodes) is **persistent** if it is active (has at least one event) in **at least 2 of the 5 windows**. We want **rho = the fraction of ALL edges in the graph that are persistent** (a number between 0 and 1).

**What you see.** A random walk observed a sample of the graph's edges. For each observed edge we recorded how many times the walk observed it (**n**) and in how many **distinct windows** those observations fell (**w**). Crucially, **w can undercount an edge's true active-window count**: if the walk observes an edge only a few times, it may land in fewer windows than the edge is actually active in. So a naive count of "observed edges with w>=2 over observed edges" is biased **downward**.


**Your task.** For each case below, estimate rho for the full graph. Reason about how much the under-observation deflates what you see, and correct for it. Then give your final answer.

**Output format.** End with a line `ANSWERS:` followed by one line per case as `<case_id>: <number between 0 and 1>`. Give all 5 cases.

---

### Case 1
- Total observed edges: 738
- Time windows: 5
- Observation histogram (cell = number of observed edges seen n times that appeared in w distinct windows):

| times seen (n) \ distinct windows (w) | w=1 | w=2 | w=3 | w=4 | w=5 |
|---|---|---|---|---|---|
| n=1 | 490 | 0 | 0 | 0 | 0 |
| n=2 | 122 | 7 | 0 | 0 | 0 |
| n=3 | 49 | 6 | 0 | 0 | 0 |
| n=4 | 31 | 3 | 1 | 0 | 0 |
| n=5 | 13 | 2 | 0 | 0 | 0 |
| n=6-10 | 12 | 0 | 0 | 0 | 0 |
| n=11+ | 1 | 1 | 0 | 0 | 0 |

### Case 2
- Total observed edges: 692
- Time windows: 5
- Observation histogram (cell = number of observed edges seen n times that appeared in w distinct windows):

| times seen (n) \ distinct windows (w) | w=1 | w=2 | w=3 | w=4 | w=5 |
|---|---|---|---|---|---|
| n=1 | 423 | 0 | 0 | 0 | 0 |
| n=2 | 141 | 5 | 0 | 0 | 0 |
| n=3 | 55 | 2 | 0 | 0 | 0 |
| n=4 | 31 | 3 | 0 | 0 | 0 |
| n=5 | 12 | 1 | 0 | 0 | 0 |
| n=6-10 | 17 | 1 | 0 | 0 | 0 |
| n=11+ | 1 | 0 | 0 | 0 | 0 |

### Case 3
- Total observed edges: 1473
- Time windows: 5
- Observation histogram (cell = number of observed edges seen n times that appeared in w distinct windows):

| times seen (n) \ distinct windows (w) | w=1 | w=2 | w=3 | w=4 | w=5 |
|---|---|---|---|---|---|
| n=1 | 957 | 0 | 0 | 0 | 0 |
| n=2 | 265 | 36 | 0 | 0 | 0 |
| n=3 | 96 | 14 | 0 | 0 | 0 |
| n=4 | 45 | 2 | 0 | 0 | 0 |
| n=5 | 23 | 2 | 1 | 0 | 0 |
| n=6-10 | 25 | 3 | 1 | 0 | 0 |
| n=11+ | 3 | 0 | 0 | 0 | 0 |

### Case 4
- Total observed edges: 1513
- Time windows: 5
- Observation histogram (cell = number of observed edges seen n times that appeared in w distinct windows):

| times seen (n) \ distinct windows (w) | w=1 | w=2 | w=3 | w=4 | w=5 |
|---|---|---|---|---|---|
| n=1 | 996 | 0 | 0 | 0 | 0 |
| n=2 | 270 | 39 | 0 | 0 | 0 |
| n=3 | 94 | 15 | 3 | 0 | 0 |
| n=4 | 37 | 2 | 1 | 0 | 0 |
| n=5 | 23 | 3 | 0 | 0 | 0 |
| n=6-10 | 21 | 4 | 1 | 0 | 0 |
| n=11+ | 3 | 0 | 1 | 0 | 0 |

### Case 5
- Total observed edges: 1398
- Time windows: 5
- Observation histogram (cell = number of observed edges seen n times that appeared in w distinct windows):

| times seen (n) \ distinct windows (w) | w=1 | w=2 | w=3 | w=4 | w=5 |
|---|---|---|---|---|---|
| n=1 | 876 | 0 | 0 | 0 | 0 |
| n=2 | 269 | 15 | 0 | 0 | 0 |
| n=3 | 106 | 0 | 0 | 0 | 0 |
| n=4 | 56 | 4 | 1 | 0 | 0 |
| n=5 | 31 | 1 | 0 | 0 | 0 |
| n=6-10 | 37 | 0 | 1 | 0 | 0 |
| n=11+ | 1 | 0 | 0 | 0 | 0 |


---
Now give your reasoning briefly, then the `ANSWERS:` block with all 5 estimates.