# Sample repo

A small project with planted, realistic bugs (no hint comments) and a test suite
that pins down the correct behaviour. Used to demo and evaluate the agent:

```bash
repo-debug-agent examples/sample_repo --test-cmd "python -m pytest -q"
```

Planted bugs (don't show this file to the agent if you want a fair test; it only
reviews source files anyway):

| File | Function | Bug | Caught by tests |
| --- | --- | --- | --- |
| `inventory.py` | `Cart.__init__` | mutable default argument shared between carts | yes |
| `inventory.py` | `Cart.total` | ignores `quantity` | yes |
| `stats.py` | `mean` | `ZeroDivisionError` on empty input (docstring says 0.0) | yes |
| `stats.py` | `median` | wrong middle elements for even-length input (IndexError / wrong value) | yes |
| `stats.py` | `moving_average` | off-by-one: drops the last window | yes |
| `textutil.py` | `word_count` | file handle never closed | no |
| `textutil.py` | `truncate` | `<` instead of `<=`: text exactly `limit` long gets cut | yes |
| `web/cart.js` | `cartTotal` | `<=` loop bound reads past the end of the array | no |

Correct code the agent should leave alone: `Cart.add`, `Cart.remove`,
`Cart.apply_discount`, `clamp`, `slugify`, `formatPrice`.
