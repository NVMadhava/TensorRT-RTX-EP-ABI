# Experimental runtime engine creation: implementation and findings

## Purpose and isolation

This work is an experiment, not a proposed production patch. It lives on the disposable branch
`codex/runtime-engine-prototype`, based on commit `1958a8b` from the main dynamic-shape/DQ investigation. The existing
session-initialization behavior remains the default. Runtime engine creation is enabled only when this environment
variable is set:

```text
ORT_TRT_RTX_EXPERIMENTAL_RUNTIME_ENGINE_BUILD=1
```

The prototype was created to answer one question with measured evidence: if TensorRT engine creation is delayed until
the first execution, when runtime shape-tensor values are available, how many of the 464 reproducible failures disappear
without CPU fallback?

## Changes made

The implementation changes two provider source files by 491 insertions and 59 deletions relative to `1958a8b`.

1. The existing TensorRT serialization block was extracted into `BuildSerializedNetworkForNode`. This refactor does not
   change when an engine is built; it makes the same operation callable from session initialization or execution.
2. In experimental mode, a graph with dynamic inputs and no user-provided profile is parsed during session
   initialization, but its TensorRT network, builder configuration, parser, and related state are retained instead of
   immediately serializing an engine.
3. On execution, the provider constructs a runtime signature from every execution tensor's dimensions and every shape
   tensor's actual values. It creates an exact TensorRT optimization profile (`min = opt = max`) for a previously unseen
   signature, serializes a replacement engine containing all signatures observed so far, and creates a new execution
   context.
4. The signature is mapped to its profile index. A later call with a known signature selects the existing profile and
   does not rebuild the engine. A new signature adds another profile and rebuilds the engine once.
5. Replacement and shutdown ordering were made explicit: the execution context is destroyed before the engine on which
   it depends. Move assignment was added to the context deleter so a rebuilt context can replace the old one safely.
6. CUDA Graph is disabled for shape-tensor graphs in this path. Compile-only and EP-context generation are rejected
   explicitly because the prototype cannot serialize an engine before it has runtime values.

User-provided profiles retain priority. Graphs that do not meet the experimental condition continue through the existing
session-time path.

## Focused validation

Four representative tests that previously stopped at the missing-profile gate passed with TensorRTRTX, without manual
profile options and without CPU fallback:

| Test | Runtime shape-tensor values observed by the prototype | Result |
|---|---|---:|
| `CumSumTest._1DTestExclusive` | `axis=[0]` | Passed |
| `ExpandOpTest.Expand_1x3` | `shape=[1,3]` | Passed |
| `PadOpTest.ConstantPadAxes` | `pads=[0,1,0,1]`, `axes=[1,3]` | Passed |
| `SliceTest.Slice1D_Regular` | `starts=[2]`, `ends=[4]`, `axes=[0]` | Passed |

A custom same-session Slice runner then exercised `end=2`, `end=4`, a negative-step reverse slice, and `end=2` again.
All four executions passed. The first three distinct signatures caused profiles to be added; the final call reused profile
0 and did not rebuild. This also corrected an earlier naive attempt that built only for the first values and returned a
stale result when later values changed.

The focused Slice inventory improved from 46 failures out of 48 baseline tests to 5 failures out of 53 selected Slice and
DynamicSlice tests. It fixed 41 of the original 46 Slice failures. The five remaining failures are outside the basic
missing-profile problem:

- one negative-step reversal semantic mismatch;
- two expected validation failures whose detailed TensorRT reason is replaced by a generic EP serialization error;
- one parser/capability limitation for a large-step case;
- one TensorRT/Myelin zero-dimension build limitation.

## Full-suite result

The same 5,533 runnable test results were compared between the DQ step-4 baseline and the runtime-engine prototype.

| Result | Baseline | Prototype | Change |
|---|---:|---:|---:|
| Passed | 4,948 | 5,229 | +281 net |
| Failed | 464 | 183 | -281 |
| Skipped | 121 | 121 | 0 |

Of the 464 original failures, 284 now pass (61.2%). Of the 183 prototype failures, 180 were already failures and 3 are
newly visible. The net reduction is 281 failures, or 60.6%. CPU fallback was not enabled.

The batch runner also listed five disabled tests as hangs because their filters ran zero tests. They are not observed
runtime hangs and are not included among the 5,533 terminal results.

### Three newly visible Tile failures

The three tests are `TileOverflowRepeats1D`, `TileOverflowRepeats2D`, and `TileOverflowRepeatsInt32`. They pass in the
baseline only because the unrelated missing-profile error causes initialization to fail, which happens to satisfy each
test's broad expectation that execution must fail. With runtime values available, TensorRT-RTX builds and runs the graph,
but does not reject repeats whose output-dimension multiplication overflows signed `int64_t`; the test then correctly
reports `Run succeeded but expected failure`.

These are not evidence that the signature cache returned a wrong profile. They expose a pre-existing Tile overflow-
validation gap that the earlier profile failure masked. A production implementation must validate this arithmetic before
building or running the TensorRT graph.

## What the result proves—and what it does not

The experiment proves that runtime engine creation is a broad solution for this inventory: it directly recovers 284 of
the 464 original failures while preserving native TensorRTRTX execution. It also proves that caching profiles by the
complete runtime signature can correctly serve different shape-tensor values in one session.

It does **not** prove that all 284 recovered tests *require* runtime engine creation. Earlier Pad and Slice experiments
showed that a prebuilt engine can serve multiple values when safe bounds and runtime validation are available. Some of
the 284 could therefore be addressed by operator-aware profile generation. The defensible count is:

- 284 failures are demonstrated to be solvable by this runtime-engine approach;
- zero of those 284 are proven to have runtime engine creation as their only possible native solution;
- the remaining 180 failures require separate classification because they passed the profile gate and exposed other
  parser, semantic, validation, type, zero-dimension, or error-propagation problems.

## Why this is not ready for production

- Every unseen exact signature adds a profile and rebuilds the complete engine. Models with many value combinations can
  incur repeated build latency and unbounded profile/engine growth.
- Rebuilding an engine to add profiles is only a prototype cache policy. A production design needs bounded caches,
  eviction, concurrency control, and a decision about whether to cache separate engines or broader validated profiles.
- Parser, network, builder configuration, initializer, engine, and execution-context lifetimes are extended beyond their
  original session-initialization scope. Multi-session and concurrent-run stress testing has not been performed.
- Existing engine cache/load/dump, timing cache, EP-context generation, compile-only mode, and CUDA Graph behavior have
  not been integrated into the delayed-build lifecycle.
- Exact profiles avoid inventing unsafe ranges but may specialize excessively. Broader profiles require operator-aware
  validity rules and explicit checks that runtime values fall inside the selected profile.
- Runtime errors currently lose useful TensorRT diagnostics in several expected-failure tests.
- The newly exposed Tile overflow gap must be fixed or rejected safely.

## Artifacts and reproducibility

- Branch: `codex/runtime-engine-prototype`
- Refactor commit: `5cbf88e`
- Runtime-signature prototype commit: `400100e`
- Full logs: `build-phase1-ninja/runtime-engine-prototype/full-suite` (untracked build artifact)
- Same-session log: `build-phase1-ninja/runtime-engine-prototype/repeated-values-signatures-v2.log` (untracked build artifact)

The test environment's original provider DLL was restored after the run. Its SHA-256 matched the retained backup:
`2EE746F4D92A90176066CE72331962FFFB1E7936A89A9615338B9F6656A0F789`.

## Recommendation

Keep this branch as experimental evidence and do not merge it into the main DQ work. The 464-to-183 result justifies a
design discussion about delayed or hybrid engine creation. Before choosing that architecture, compare it against a
hybrid policy: build at session initialization when profiles are available or can be generated safely, and delay only
graphs whose value-dependent shape inputs cannot be represented soundly. That discussion should include cache bounds,
concurrency, error semantics, and the behavior of engine/EP-context caches.
