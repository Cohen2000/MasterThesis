"""Generic current zero-shot prompt; no historical experiment imports."""

COMPONENTS = ("rho_2", "rho_3", "rho_4", "rho_5")


def build_prompt(observed_data: str, *, selection_access: str,
                 history_access: str) -> str:
    """Render explicit observation semantics supplied by the sampling caller.

    Access descriptions must state the actual selection and history rules,
    including budgets/cutoffs when set. This renderer does not invent them.
    """
    if any(not isinstance(value, str) or not value.strip()
           for value in (observed_data, selection_access, history_access)):
        raise ValueError("observed data and both access descriptions are required")
    return f"""TARGET / DEFINITIONS
The complete temporal graph is normalized to [0,1) and divided into W=5 equal,
non-overlapping windows: [0,0.2), [0.2,0.4), [0.4,0.6), [0.6,0.8), [0.8,1).
E_full consists of every undirected dyad with at least one interaction event
in the complete stream. For each dyad e, A_e(w)=1 if e has at least one event
in window w and is 0 otherwise. K_e is the sum of A_e(w) over the five windows.
For k=2,3,4,5, rho_k is the number of dyads in E_full with K_e>=k divided by
the number of dyads in E_full. Each dyad contributes equally to this target.
The predictions must satisfy 1 >= rho_2 >= rho_3 >= rho_4 >= rho_5 >= 0.

OBSERVATION MECHANISM
Selection/access mechanism:
{selection_access.strip()}
History access:
{history_access.strip()}

OBSERVED DATA
{observed_data.strip()}

TASK
Estimate rho_2, rho_3, rho_4 and rho_5 jointly for the complete temporal graph.
You may reason freely before the final result.

FINAL JSON RESULT
The final non-empty line must be one JSON object containing only the keys
rho_2, rho_3, rho_4, rho_5, each with a numeric value. Do not put that line in
a Markdown code fence or add text after it."""
