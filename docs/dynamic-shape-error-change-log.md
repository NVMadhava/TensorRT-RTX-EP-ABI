# Dynamic-shape error investigation and change log

## Purpose

This file is the merge reference for the dynamic-shape failures found while running the ONNX Runtime provider tests against the TensorRT RTX EP ABI. It records what was changed, why it was changed, what was deliberately not changed, and the exact test result after each step.

The original test handoff is `C:\Users\amadhavasrir\Downloads\bulding_files\results\AGENT_HANDOFF_TEST_ERRORS.md`.

## Baseline and current scope

- Investigation date: 2026-08-17
- TensorRT RTX EP ABI baseline commit: `7bc9add`
- ONNX Runtime baseline: 1.29, commit `2e2543f`
- TensorRT RTX version: 1.7.0.56
- CUDA version: 13.4
- Current focus: `ExpandOpTest.Expand_1x3` and `CumSumTest._1DTestExclusive`
- Agreed design boundary: do not introduce runtime/lazy engine creation in this series. That is a much larger lifecycle change and requires separate design discussion.

## Version-control and integration index

Working branch: `codex/ep-abi-bug-fixes`

The Git commits are the authoritative code history. This document records the reasoning, evidence, limitations, and recommended integration order. Generated files under `build-phase1-ninja/` are deliberately not included in any commit.

| Order | Commit | Area | Files | Verification status | Integration status |
|---:|---|---|---|---|---|
| 1 | `d590d0f` | Shape-tensor profile diagnostics and bounds validation | `src/tensorrt_rtx_execution_provider.cc` | Targeted Expand and CumSum investigation completed; CumSum access violation replaced by a controlled initialization failure | Memory-safety portion is a merge candidate; diagnostic log level/noise should be reviewed before final merge |
| 2 | `0c380a0` | Runtime scalar Q/DQ arithmetic lowering and graph-level tests | `src/qdq_lowering.cc`, `tests/test_tensorrt_rtx_proto_preprocessing.cpp` | Focused tests passed; a complete batched run produced 4,941 pass / 471 fail / 121 skip, with zero new failures versus baseline | Runtime scalar DQ has full-suite regression evidence; general runtime Q remains experimental because exact rounding equivalence is not yet proven |
| 3 | documentation commit | Investigation history and integration index | `docs/dynamic-shape-error-change-log.md` | Documentation review | Merge with or after the code commits |

### Recommended integration procedure

1. Review and cherry-pick `d590d0f` independently. If permanent warning-level diagnostics are considered too noisy, reduce or remove only the diagnostic statements while retaining the profile-length guard.
2. Review `0c380a0` separately. Do not treat passing UINT8 examples as proof that runtime QuantizeLinear is universally safe; add rounding-boundary coverage before approving the Q portion as a general fix.
3. Build the newly added preprocessing tests in a test-enabled build and run them before final integration.
4. Cherry-pick the documentation commit so the reasoning and known limitations remain beside the implementation history.

These commits are ordered but do not depend on each other at the source level: the Q/DQ commit can be reviewed, reverted, or deferred without removing the shape-profile safety fix.

## Phase 1: diagnostic logging

### Status

Implemented and tested. This phase adds observations only; it does not intentionally change provider behavior.

### File changed

`src/tensorrt_rtx_execution_provider.cc`

### Diagnostic additions and reasons

All new messages contain `[NvTensorRTRTX EP][DynamicProfileDiag]` so they can be isolated from the normal provider log.

1. **Compile input** logs the input name, whether TensorRT classifies it as a shape tensor, its rank (`nb_dims`), and its dimensions. This establishes how TensorRT sees an input before an optimization profile is constructed.
2. **Implicit profile constructed** logs the number and values of the generated min/opt/max entries and the `has_implicit_profile` flag. This shows whether the provider generated values for the tensor and whether it considered that generated profile usable.
3. **Applying shape-tensor profile** logs the calculated number of required values and the actual min/opt/max vector lengths immediately before the values are read. This was added to find malformed vectors and the location of the CumSum access violation.
4. **Final profile gate** logs `has_dynamic_shape`, `has_explicit_profile`, `has_implicit_profile`, and the TensorRT profile count. This explains why the provider either applies a profile or falls through to the exact-shape build path.

### Build artifact

- Build directory: `build-phase1-ninja`
- Built library: `build-phase1-ninja\onnxruntime_providers_nv_tensorrt_rtx.dll`
- SHA-256: `4FD941FC3AE3BDC51252ABECFC156285287079ACEEBC88F8E47057DAA8CDAD5A`
- Isolated staged copy: `C:\Users\amadhavasrir\Downloads\bulding_files\test-env\onnxruntime_providers_nv_tensorrt_rtx.phase1-diag.dll`
- The active test-environment DLL was not overwritten.

### Test evidence

#### `ExpandOpTest.Expand_1x3`

Result: failed with a controlled initialization error, not a process crash.

- `data_0`: ordinary tensor, dimensions `[3,1]`, implicit entries `[3/3/3,1/1/1]`.
- `data_1`: shape tensor, dimensions `[2]`, but the implicit builder generated one entry with values `[2/2/2]`.
- Final flags: dynamic shape `true`, explicit profile `false`, implicit profile `false`, TensorRT profile count `1`.

Interpretation: for `Expand`, `data_1` has container shape `[2]`, while its runtime contents are the requested output shape `[1,3]`. The current implicit builder uses the container dimension `2` as a profile value. It therefore does not describe the runtime contents required by TensorRT.

#### `CumSumTest._1DTestExclusive`

Result: failed with Windows structured exception `0xC0000005` (invalid memory access).

- `x`: ordinary tensor, dimensions `[5]`.
- `axis`: scalar shape tensor, rank `0`, dimensions `[]`.
- The implicit builder generated zero min/opt/max entries for `axis`.
- The application path calculated that a scalar shape tensor requires one value, then attempted to read element `0` from each empty vector.

Interpretation: the immediate memory-safety bug is a missing length check before indexing the profile vectors. This is separate from deciding what valid profile range should be generated for a runtime CumSum axis.

## Phase 2A: reject incomplete shape-tensor profiles safely

### Intended scope

Add a bounds/consistency check before `ApplyProfileShapesFromProviderOptions` reads min/opt/max values for a shape tensor. A scalar TensorRT shape tensor requires one profile value even though its tensor rank is zero. If the implicit profile vectors contain zero values, the provider must report that the profile cannot be applied instead of indexing outside the vectors.

This is deliberately a memory-safety fix only:

- Expected: `CumSumTest._1DTestExclusive` no longer terminates with `0xC0000005`.
- Expected for now: the test still fails with a controlled provider initialization error because no valid runtime-axis profile has yet been generated.
- Not included: runtime engine creation, reading the runtime axis during compilation, or selecting final CumSum axis bounds.

### Implementation and verification

Implemented and verified for the isolated failing subtest.

#### Source change

`src/tensorrt_rtx_execution_provider.cc`, in `ApplyProfileShapesFromProviderOptions`:

- Store references to the min/opt/max value vectors used by the following loop.
- Calculate the number of values that TensorRT will read.
- Require all three vectors to have exactly that length.
- Log the required and actual lengths and return `false` when they do not match.
- Only index the vectors after the check succeeds.

The following source comment was added with the guard because it captures the non-obvious reason that rank zero does not mean zero profile values:

```cpp
// A scalar shape tensor has nbDims == 0, but TensorRT still requires one
// profile value. Reject incomplete profiles before indexing their vectors.
```

Why exact equality is checked: a profile is valid only when every min/opt/max vector describes the same number of shape-tensor values that will be passed to TensorRT. Fewer values are unsafe to read; extra or inconsistent values indicate malformed profile data and should not be silently ignored.

#### Build artifact

- Incremental build result: successful
- Built library: `build-phase1-ninja\onnxruntime_providers_nv_tensorrt_rtx.dll`
- File size: 2,312,192 bytes
- SHA-256: `FDFC271A37EEBD96F5E56E7B3497DF64CE07865C8FAE734083D4324A84C8FC46`
- Isolated staged copy: `C:\Users\amadhavasrir\Downloads\bulding_files\test-env\onnxruntime_providers_nv_tensorrt_rtx.phase2a-safety.dll`
- The Phase 1 staged DLL and the active test-environment DLL were not overwritten.

#### Targeted test

Command target: `CumSumTest._1DTestExclusive`

Result: the process completed the Google Test run normally in 152 ms. The test still reports failed because it expected successful model initialization, but there was no `0xC0000005` structured exception.

The new guard reported:

```text
Cannot apply shape-tensor optimization profile for input 'axis':
required_value_count=1, min_value_count=0, opt_value_count=0, max_value_count=0
```

The failure presented to the test was:

```text
Initialize failed but expected success: Optimization profile could not be applied for tensor:
axis
[]
```

Conclusion: **the invalid memory access is resolved for `CumSumTest._1DTestExclusive`**. Phase 2A meets its memory-safety goal. CumSum runtime-axis support is not functionally fixed yet; the next separately reviewed change must decide how to construct a valid axis profile or whether such graphs should be left unclaimed by this provider.

The later `Failed to find symbol GetProvider ... error code: 127` log is the known legacy-provider probing message from the test infrastructure and is not the CumSum failure being addressed here.

## Phase 2B: CumSum runtime-axis feasibility experiments

### Scope

These were isolated experiments only. No additional provider source behavior was changed. Experimental models and a standalone runner were placed under the untracked build directory `build-phase1-ninja\cumsum-rewrite-experiment`. Every rewrite run disabled CPU fallback, so successful inference means the complete experimental graph was assigned to TensorRT RTX.

### Fixed and variable profile experiments on the original CumSum graph

For rank-1 CumSum, the following explicit axis profiles were tested through `default_ep_options`:

| Axis profile (min/opt/max) | Runtime axis | Result |
|---|---:|---|
| no values | `0` | Controlled initialization failure; Phase 2A prevented the access violation |
| `0/0/0` | `0` | Passed |
| `-1/0/0` | `0` | Engine build failed |
| `-1/-1/-1` | `-1` | Passed |

The variable profile failed with TensorRT RTX reporting:

```text
ICumulativeLayer 'axis' value must be a build time constant
```

Conclusion: negative axes are supported, but each TensorRT CumSum layer requires one fixed axis while the engine is built. A legal range such as `[-R, R-1]` cannot be applied directly to the original CumSum axis input.

### Single-engine fixed-branch rewrite

The experiment replaced one runtime-axis CumSum with one fixed-axis CumSum per tensor rank, normalized negative axes, and selected the appropriate result at runtime. For example, rank 3 contains fixed CumSum branches for axes `0`, `1`, and `2`.

#### Rank-2 standard CumSum

Input shape: `[2,3]`.

| Runtime axis | Expected normalized axis | Result |
|---:|---:|---|
| `0` | `0` | Passed |
| `-2` | `0` | Passed |
| `1` | `1` | Passed |
| `-1` | `1` | Passed |

Both a chained `Where` selector and a stacked-results `Gather` selector produced correct results for all legal axes.

#### Rank-2 reverse and exclusive CumSum

The `Gather` form was also tested with both `reverse=1` and `exclusive=1`. Runtime axes `0`, `-2`, `1`, and `-1` all produced the expected results.

#### Rank-3 standard CumSum

Input shape: `[2,2,3]`.

| Runtime axis | Expected normalized axis | Result |
|---:|---:|---|
| `0` | `0` | Passed |
| `-3` | `0` | Passed |
| `1` | `1` | Passed |
| `-2` | `1` | Passed |
| `2` | `2` | Passed |
| `-1` | `2` | Passed |

This confirms that the rewrite concept is not limited to rank 2.

### Unresolved correctness issue: invalid axes

Invalid positive axes were tested for both ranks:

| Input rank | Invalid runtime axis | Observed result |
|---:|---:|---|
| 2 | `2` | No error; returned an all-zero output tensor |
| 3 | `3` | No error; returned an all-zero output tensor |

This is not ONNX-compatible behavior. The original CumSum operator is required to reject an out-of-range axis. Replacing it with `Where` or `Gather` selection therefore cannot be accepted as a complete implementation unless invalid-axis validation is added.

After the rewrite, TensorRT classified the runtime `axis` input as an execution tensor rather than a shape tensor. The current profile application path did not enforce the supplied scalar value range for that rank-zero execution tensor, so the `[-R, R-1]` provider profile did not reject invalid inputs.

### Current assessment

The fixed-branch rewrite is technically feasible for valid runtime axes, including negative axes, rank 2, rank 3, and combined reverse/exclusive behavior. It preserves a single compile-time-created TensorRT engine. It is not ready for implementation because invalid-axis behavior is incorrect and because computation/memory cost grows with rank: a rank-`R` input creates `R` CumSum branches.

## Entry format for later work

Every later change should add: the affected error/test, root cause, exact source files and behavior changed, why the chosen fix is appropriately scoped, alternatives intentionally deferred, build artifact/hash, tests run, and remaining limitations.

## Quantization Phase 1: runtime scalar Q/DQ parameter lowering

### Affected failures

The two initial targets were:

- `DequantizeLinearOpTest.Int8`
- `QuantizeLinearOpTest.2D`

Both originally failed during session initialization. TensorRT's parser accepted the native Q/DQ node during capability discovery, so the EP claimed it. The stricter engine validator then rejected it with:

```text
IQuantizeLayer/IDequantizeLayer node1 has `zero_point` which is not an initializer
```

The test inputs `scale` and `zero_point` are runtime graph inputs rather than ONNX initializers. The existing Q/DQ lowering pass required a readable scale initializer and therefore copied both nodes through unchanged.

### Source behavior changed

`src/qdq_lowering.cc` now recognizes a deliberately narrow runtime-parameter form:

- scale and zero point must have statically known scalar or one-element shapes;
- scale must be `float32`;
- runtime `DequantizeLinear` lowering accepts matching `int8` or `uint8` input/zero-point types;
- runtime `QuantizeLinear` lowering currently accepts `uint8` output only.

For runtime DQ, the native node is replaced with:

```text
Cast(x to float32) -> Sub(Cast(zero_point to float32)) -> Mul(scale)
```

For runtime UINT8 Q, the native node is replaced with:

```text
Cast(x) -> Div(scale) -> Add(Cast(zero_point)) -> Round
        -> Max(qmin) -> Min(qmax) -> Cast(uint8)
```

The preprocessing entry point already runs before both capability discovery and engine creation, so this change does not move engine creation, add runtime compilation, or alter the overall EP pipeline.

The following explanatory comments were added to the implementation:

- why Phase 1 accepts only scalar/one-element runtime parameters;
- why runtime scale positivity cannot be inspected at compile time and remains an ONNX caller contract;
- why runtime INT8 QuantizeLinear is intentionally excluded pending exact rounding behavior.

`tests/test_tensorrt_rtx_proto_preprocessing.cpp` now contains graph-level coverage asserting that runtime scalar Q/DQ nodes are removed, the expected arithmetic nodes are emitted, the original output name is preserved, and the original ORT node id in `doc_string` remains attached. The current Ninja build cache has `BUILD_TESTS=OFF`, so these two newly added source-level tests were not built in this run; the provider integration tests below exercised the production preprocessing path in both capability and compile.

### Numerical boundary experiment and resulting scope reduction

An intentionally broader first draft also lowered runtime scalar INT8 QuantizeLinear. `QuantizeLinearOpTest.Int8` then built and ran, but returned `-128` for one element where ORT expects `-127`.

The relevant FP32 values are:

```text
scale stored as float32 = 0.039215688
-5 / scale in ORT-compatible float32 arithmetic = -127.49999
```

ORT therefore rounds this value to `-127`. TensorRT's optimized arithmetic graph produced the boundary value as `-127.5` and rounded ties-to-even to `-128`. Promoting the emitted arithmetic to FP64 was tested, but the TensorRT/Myelin backend still produced `-128`; that experiment was removed.

To avoid introducing a silent numerical error, runtime INT8 Q lowering was excluded from the final Phase 1 patch. Its behavior remains the original controlled engine-build failure rather than an incorrect successful result. Runtime UINT8 Q and INT8/UINT8 DQ remain enabled because the selected tests and additional controls passed.

### Build artifact

```text
Path: C:\Users\amadhavasrir\Downloads\bulding_files\test-env\onnxruntime_providers_nv_tensorrt_rtx.runtime-qdq-scalar.dll
Size: 2313728 bytes
SHA256: 5B23D560110834E087F89C6390B3504D404E7504ADD8A21FA9CAB90112B09C3A
```

The artifact was copied under a separate name; the baseline provider DLL was not overwritten.

### Tests run

Focused tests with `nv_detailed_build_log=1`:

| Test | Result | Provider-path evidence |
|---|---|---|
| `DequantizeLinearOpTest.Int8` | Passed | TensorRT parsed the emitted Cast/Sub/Mul graph and built a Myelin engine with all three runtime inputs |
| `QuantizeLinearOpTest.2D` | Passed | TensorRT parsed the emitted Div/Add/Round/Max/Min/Cast graph and built a Myelin engine with all three runtime inputs |

Additional controls:

| Test | Result |
|---|---|
| `DequantizeLinearOpTest.Int8_Large` | Passed |
| `DequantizeLinearOpTest.Int8_NonAlignedSize_Initializer` | Passed; existing initializer path preserved |
| `QuantizeLinearOpTest.Uint8` | Passed |
| `QuantizeLinearOpTest.Scalar` | Passed |
| `QuantizeLinearOpTest.Int8` | Still fails at engine creation by design; unsafe arithmetic result is no longer used |

### Complete batched provider-suite regression

The complete provider inventory was rerun against the Phase 1 DLL in batches, using the same filter as the original baseline:

```text
-*NvExecutionProviderTest*
```

Result directory:

```text
C:\Users\amadhavasrir\Downloads\bulding_files\results\runtime-qdq-full-20260817-152243
```

The runner enumerated 5,538 names. Five names contain Google's `DISABLED_` marker and therefore intentionally execute zero tests unless `--gtest_also_run_disabled_tests` is supplied. Excluding those five, the result inventory exactly matches the 5,533 enabled-test baseline.

| Result | Baseline | Runtime scalar Q/DQ DLL | Change |
|---|---:|---:|---:|
| Passed | 4,930 | 4,941 | +11 |
| Failed | 482 | 471 | -11 |
| Skipped | 121 | 121 | 0 |
| Enabled total | 5,533 | 5,533 | 0 |

Failure-set comparison:

- new failures relative to the baseline: **0**;
- baseline failures now passing: **11**;
- genuine hangs or unaccounted enabled tests: **0**.

The five disabled names appear in `hangs.txt` because the existing resume script treats a zero-test batch as "no progress." Their individual logs say `Running 0 tests` and `YOU HAVE ... DISABLED TESTS`; they were not process hangs or crashes.

The 11 newly passing tests divide into five DQ tests, five Q tests, and one shape-profile/CumSum test.

Runtime-scalar DQ accounts for these five improvements:

```text
DequantizeLinearOpTest.DequantizeLinear_per_tensor_float_int8
DequantizeLinearOpTest.Int8
DequantizeLinearOpTest.Int8_Large
DequantizeLinearOpTest.Scalar
DequantizeLinearOpTest.Zero_Point_int8
```

Across the complete inventory, 82 executed test names contain `DequantizeLinear`: 75 passed and 7 failed. All seven failures were already present in the 482-failure baseline, so the DQ change introduced no observed DQ regression:

```text
DequantizeLinearOpTest.Int4_LargeInitializerInput
DequantizeLinearOpTest.Int4NoZeroPoint
DequantizeLinearOpTest.Without_Zero_Point
DequantizeLinearOpTest.Per_Channel_Axis_Default
DequantizeLinearOpTest.Per_Channel_Axis_1_int8
DequantizeLinearContribOpTest.DequantizeLinear_1
DequantizeLinearContribOpTest.DequantizeLinear_2
```

These remaining failures do not exercise the newly supported runtime scalar `float32` scale plus matching `int8`/`uint8` zero-point form. They cover forms such as INT4, omitted zero point, per-channel parameters, and separate contrib cases.

The full-suite result materially increases confidence in runtime scalar DQ: it fixes five baseline failures and produces no new failures across the enabled inventory. It does not prove behavior for arbitrary invalid runtime scale values because a runtime value cannot be checked during graph preprocessing; ONNX still requires scale to be positive.

### Remaining limitations and next work

- Runtime INT8 QuantizeLinear needs an exact rounding strategy before it can use arithmetic lowering.
- Per-channel runtime scale/zero-point tensors are not lowered yet; they require axis-aware runtime reshape/broadcast handling.
- Scalar Q/DQ without a zero-point still follows the native TensorRT path and retains the separate scalar-rank `{}` to `{1}` issue.
- FP16/BF16 scales, INT16/UINT16, block quantization, INT4, and Float8 remain outside this phase.
