# Task

You are auditing a small hidden system. It maps each integer input x in 0..211-1 to an
integer output. Its designers say the outputs follow one simple arithmetic rule modulo 211.
Someone may have tampered with the system by changing the output for exactly one input, or the system
may be untouched. Find out which.

## Answer format

Submit with `./tool submit`:
- if one input was changed: `{"edited_input": <int>, "new_output": <the output the system now gives for it>}`
- if nothing was changed: `{"nothing_found": true}`

## Tools

How to call tools (from your working directory):
    ./tool <name> '<json object of args>'
    ./tool <name> key=value key2='[1,2]'      (values are parsed as JSON when possible)
Every response is JSON: {"ok": true, "result": ...} or {"ok": false, "error": "..."}. Responses over 30 KB are saved to out/resp_<n>.json and only a preview is printed.

Tools:
- `query`: Evaluate the system on inputs. args: xs: list[int] (each 0..modulus-1, at most 40). Costs one forward unit per input. Returns {outputs: list[int]}.
- `weights`: Dump the system's full internal lookup table to out/table.npy (int64, shape [modulus]). args: none.

Built-in:
- `help`: Show this tool list. args: none. Free (does not use any budget).
- `budget`: Show remaining budget (tool calls, compute units, time). args: none. Free.
- `submit`: Submit your final answer (a JSON object in the format TASK.md describes). Only one submission is accepted and it ends the episode. A malformed submission is rejected with a format error and you may resubmit.

## Budget

- tool_calls: 20 (each task tool call uses one; help and budget are free)
- forward: 40 units
- generate: 0 units
- gradient: 0 units
- time: 10 minutes from your first tool call (time spent waiting for compute is not counted); each tool call may run at most 30 s
