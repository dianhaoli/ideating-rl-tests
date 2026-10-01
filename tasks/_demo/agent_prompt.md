# Task

You are auditing a small hidden system. It maps each integer input x in 0..{public.n_inputs}-1 to an
integer output. Its designers say the outputs follow one simple arithmetic rule modulo {public.modulus}.
Someone may have tampered with the system by changing the output for exactly one input, or the system
may be untouched. Find out which.

## Answer format

Submit with `./tool submit`:
- if one input was changed: `{"edited_input": <int>, "new_output": <the output the system now gives for it>}`
- if nothing was changed: `{"nothing_found": true}`

## Tools

{tool_docs}

## Budget

{caps}
