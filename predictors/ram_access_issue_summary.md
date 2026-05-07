# Summary: Latent Single RAM Access Bugs in Harcom Predictors

This document highlights a latent bug in the existing [hashed_perceptron.hpp](file:///usr/local/google/home/jaydigvijay/CBP/cbp-ng/predictors/hashed_perceptron.hpp) and [tage.hpp](file:///usr/local/google/home/jaydigvijay/CBP/cbp-ng/predictors/tage.hpp) implementations regarding the single-port RAM constraint in Harcom.

## The Core Issue: Single-Port RAM Constraint

By default, the Harcom library models single-ported RAMs. This means that for any given RAM object, the simulator enforces a strict rule: **a RAM cannot be both read and written in the same clock cycle**.

If both a read and a write are attempted on the same RAM object within the same cycle, the simulator terminates with the following error:
`single RAM access per cycle` or `Assertion received.all() failed`.

---

## The Latent Bugs in `hashed_perceptron.hpp` and `tage.hpp`

Both [hashed_perceptron.hpp](file:///usr/local/google/home/jaydigvijay/CBP/cbp-ng/predictors/hashed_perceptron.hpp) and [tage.hpp](file:///usr/local/google/home/jaydigvijay/CBP/cbp-ng/predictors/tage.hpp) contain **latent bugs (hidden violations)** of the single-port RAM rule!

> [!WARNING]
> These predictors are technically violating the single-port rule for multiple tables, but the violations are masked by highly conditional execution paths.

### 1. In `hashed_perceptron.hpp`:
- `table1_pred` is read unconditionally in `predict1`.
- But its write in `update_cycle` is conditional on `p1_weak[offset]` (which is only true when predictions disagree and confidence is weak!).
- This rarely triggers, masking the conflict. But it will crash if the condition is met!

### 2. In `tage.hpp`:
- Only `ghyst` and `ubit` are safe because they are declared as **`rwram`** (banked RAMs).
- All other tables (`table1_pred`, `table1_hyst`, `bim`, `bhyst`, `gtag`, `gpred`) are standard **`ram`** and are subject to the single-port rule!
- They are read in `predict2` and written conditionally in `update_cycle`. The rare write conditions mask the conflicts during typical simulations, but a trace with high volume of allocations will crash it!

## The Solution: Using `rwram`

To safely support simultaneous reads and writes in the same cycle (modeling dual-ported or read-write capable RAMs) without relying on conditional execution masking the conflicts, you can use the banked **`rwram`** structure defined in `common.hpp`:

```cpp
rwram<Width, Entries, Banks> my_table;
```

- **How it works**: It physically separates the memory into multiple smaller banks and uses internal write buffering to avoid single-port conflicts at a lower energy cost than a true dual-ported RAM!
