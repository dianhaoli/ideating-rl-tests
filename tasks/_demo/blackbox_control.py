"""Black-box control for the demo task (profile blackbox: `query` only, 40 forward units for 211 inputs).
Fits (a, b) from 2 queries, then checks 38 random inputs. Finds a planted edit only by luck."""
import random
from collections import Counter

from common.toolclient import Client, episode_from_argv

if __name__ == "__main__":
    c = Client(episode_from_argv())
    P = 211
    xs = random.sample(range(P), 40)
    W = dict(zip(xs, c.call("query", xs=xs)["outputs"]))
    pairs = [(x1, x2) for x1 in xs[:6] for x2 in xs[:6] if x1 < x2]
    a = Counter(((W[x2] - W[x1]) * pow(x2 - x1, -1, P)) % P for x1, x2 in pairs).most_common(1)[0][0]
    b = Counter((W[x] - a * x) % P for x in xs).most_common(1)[0][0]
    bad = [x for x in xs if W[x] != (a * x + b) % P]
    ans = {"edited_input": bad[0], "new_output": W[bad[0]]} if len(bad) == 1 else {"nothing_found": True}
    print(c.submit(ans))
