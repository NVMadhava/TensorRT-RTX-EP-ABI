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

## DQ follow-up Step 1: scalar runtime DQ without zero point

Branch: `codex/dq-remaining-fixes`

### Root cause

`DequantizeLinearOpTest.Without_Zero_Point` has rank-zero `x`, `scale`, and `y`, with no zero-point input. The first runtime-scalar lowering required a runtime zero point, so this form remained native. TensorRT inserted an internal reshape and returned `y` with shape `{1}` instead of the ONNX scalar shape `{}`.

### Source change

`src/qdq_lowering.cc` now lowers only the following missing-zero-point form:

- `x` has a statically known scalar shape;
- `scale` is a runtime scalar/one-element FP32 tensor;
- `x` is INT8 or UINT8;
- zero point is absent.

The emitted graph is `Cast(x to FP32) -> Mul(scale)`. This is exactly `(x - 0) * scale` and preserves the scalar output rank. A graph-level preprocessing test was added to `tests/test_tensorrt_rtx_proto_preprocessing.cpp`; the current build cache still has `BUILD_TESTS=OFF`, so integration tests provide the executed-path evidence for this step.

### Artifact and tests

```text
Artifact: C:\Users\amadhavasrir\Downloads\bulding_files\test-env\onnxruntime_providers_nv_tensorrt_rtx.dq-step1-scalar-no-zp.dll
SHA256: 46D5AE62E2FB95C7D32A71DAEFA3A904CB76B6EF160FF2FA5D30E2EA3F04925B
```

Target and controls all passed:

```text
DequantizeLinearOpTest.Without_Zero_Point
DequantizeLinearOpTest.Scalar
DequantizeLinearOpTest.No_Zero_Point_int8
DequantizeLinearOpTest.No_Zero_Point_uint8
DequantizeLinearOpTest.Int8
```

Detailed TensorRT logging confirmed that the target was rewritten to Cast/Mul, built as a Myelin engine, and retained `x (Int8[]) -> y (Float[])`; it was not a CPU-fallback pass.

### Remaining limitations and next work

- Runtime INT8 QuantizeLinear needs an exact rounding strategy before it can use arithmetic lowering.
- Per-channel runtime scale/zero-point tensors are not lowered yet; they require axis-aware runtime reshape/broadcast handling.
- Scalar Q/DQ without a zero-point still follows the native TensorRT path and retains the separate scalar-rank `{}` to `{1}` issue.
- FP16/BF16 scales, INT16/UINT16, block quantization, INT4, and Float8 remain outside this phase.

## DQ follow-up Step 2: runtime per-axis INT8/UINT8 parameters

Branch: `codex/dq-remaining-fixes`

### Root cause

Four remaining per-channel tests provide both `scale` and `zero_point` as runtime rank-one tensors. The Phase 1 gate accepted runtime parameters only when they were scalar, so TensorRT received the original `DequantizeLinear`. Its native Q/DQ importer requires `zero_point` to be an initializer and rejected these graphs.

The first Step 2 diagnostic build also exposed a separate TensorRT restriction: a runtime UINT8 zero-point tensor cannot be passed directly through `Reshape`. Casting the zero point before reshaping is required.

### Source change

`src/qdq_lowering.cc` now recognizes a deliberately narrow runtime per-axis DQ form:

- `scale` and `zero_point` are both runtime rank-one tensors with the same known length greater than one;
- input rank and dimensions are static;
- the normalized ONNX `axis` is valid, and the parameter length exactly matches that input dimension;
- scale is FP32;
- input and zero point have the same INT8 or UINT8 type;
- `block_size` is absent or no greater than one.

For a non-trailing axis, scale is reshaped to an ONNX-broadcastable rank-aligned shape such as `{1, C, 1, 1}`. The zero point is first cast to FP32 and then reshaped, avoiding TensorRT's UINT8-Reshape rejection. The final equivalent graph is Cast/Reshape/Sub/Mul. A graph-level preprocessing test was added for a rank-four, axis-one case.

This does not broaden runtime QuantizeLinear handling, dynamic input dimensions, block quantization, mismatched parameter lengths/types, or non-FP32 scales.

### Artifact and tests

```text
Artifact: C:\Users\amadhavasrir\Downloads\bulding_files\test-env\onnxruntime_providers_nv_tensorrt_rtx.dq-step2-per-axis-final.dll
SHA256: 4AA2193F268E44F69BDCED7232580E68ADA4CA8DF563A5D3DADF90CF969121C2
```

All four previously failing targets passed:

```text
DequantizeLinearOpTest.Per_Channel_Axis_Default
DequantizeLinearOpTest.Per_Channel_Axis_1_int8
DequantizeLinearContribOpTest.DequantizeLinear_1
DequantizeLinearContribOpTest.DequantizeLinear_2
```

The target run contained zero `Invalid Node` and zero `No graph will run on TensorRT execution provider` messages, confirming that their success was not caused by graph rejection followed by CPU execution.

These controls also passed:

```text
DequantizeLinearOpTest.Per_Channel_Axis_0
DequantizeLinearOpTest.Per_Channel_Neg_2
DequantizeLinearOpTest.Per_Channel_Axis_1_int32
DequantizeLinearOpTest.Without_Zero_Point
```

The INT32 control retains its pre-existing native TensorRT rejection/CPU path; Step 2 intentionally does not change that form.

## DQ follow-up Step 3: odd-length INT4 without zero point

Branch: `codex/dq-remaining-fixes`

### Root cause and rejected experiments

`DequantizeLinearOpTest.Int4NoZeroPoint` has a rank-one INT4 input with five logical elements, a runtime scalar FP32 scale, and no zero point. TensorRT's ONNX Q/DQ importer rejects an odd number of 4-bit elements before engine creation.

Two smaller-looking approaches were tested and discarded:

1. Adding an explicit scalar INT4 zero point did not help. TensorRT rejected both the original explicit-zero test and the synthesized-zero graph with `4-bit quantization requires an even number of elements`. The restriction is on Q/DQ input volume, not on omission of zero point.
2. Replacing DQ by `Cast(INT4 to FP32) -> Mul(scale)` passed ONNX parsing but failed TensorRT engine validation. TensorRT does not allow an INT4 tensor to be consumed by a Cast layer.

A third padding experiment using a one-element INT4 zero initializer also failed: TensorRT cannot materialize a tensor whose storage size is only four bits and asserts that its tensor storage size must be byte-aligned.

None of these rejected experiments remains in the committed source.

### Source change

For only a statically known, rank-one, odd-length INT4 `DequantizeLinear` with runtime scalar FP32 scale and no zero point, preprocessing now emits:

```text
Concat(x, x) -> DequantizeLinear(scale) -> Slice(first N values)
```

Concatenating the vector with itself changes its logical length from odd `N` to even `2N`, which satisfies TensorRT's native INT4 DQ restriction. Dequantization is elementwise, so the first `N` outputs are exactly the outputs of the original model; Slice removes the duplicate half.

The gate intentionally excludes multidimensional inputs, dynamic lengths, even lengths, explicit zero points, non-INT4 input, and non-FP32/non-scalar scales. A preprocessing test checks that the rewrite contains exactly one Concat, one DequantizeLinear, and one Slice and preserves the output identity metadata.

### Tradeoff

This path temporarily represents `2N` quantized values and `2N` dequantized values before slicing. That can increase temporary work and memory for large odd vectors. The restriction to the exact unsupported rank-one form prevents this cost from affecting normal INT4 DQ graphs. TensorRT's optimizer fused the tested three-node graph into one Myelin engine with zero reported activation memory, but that observation is not a general guarantee for all vector sizes.

### Artifact and tests

```text
Artifact: C:\Users\amadhavasrir\Downloads\bulding_files\test-env\onnxruntime_providers_nv_tensorrt_rtx.dq-step3-int4-duplicate-slice.dll
SHA256: 03A47047D5046A213CEEDE1B85AD9251FA196D78026AFA44E9570AA99202043E
```

The target and six regression controls passed:

```text
DequantizeLinearOpTest.Int4NoZeroPoint
DequantizeLinearOpTest.Without_Zero_Point
DequantizeLinearOpTest.Int8
DequantizeLinearOpTest.Per_Channel_Axis_Default
DequantizeLinearOpTest.Per_Channel_Axis_1_int8
DequantizeLinearContribOpTest.DequantizeLinear_1
DequantizeLinearContribOpTest.DequantizeLinear_2
```

The combined run contained zero `Invalid Node` and zero `No graph will run on TensorRT execution provider` messages. Detailed logging for the target showed one Myelin engine with `x (Int4[5]), x_scale (Float[]) -> y (Float[5])`, confirming TensorRT execution rather than CPU fallback.

## DQ follow-up Step 4: INT4 initializer with runtime zero point

Branch: `codex/dq-remaining-fixes`

### Root cause

`DequantizeLinearOpTest.Int4_LargeInitializerInput` has a static rank-one INT4 initializer with 1,024 elements, a runtime scalar FP32 scale, and a runtime scalar INT4 zero point. TensorRT accepts the even-length INT4 data and runtime scale, but its native DQ layer requires the third `zero_point` input to be an initializer. Engine validation therefore failed with `zero_point which is not an initializer`.

The runtime zero point cannot safely be replaced by a compiled constant because its value is supplied for each execution. Directly casting INT4 to FP32 is also unavailable, as established in Step 3.

### Source change

The rewrite uses the exact algebraic identity:

```text
(x - zero_point) * scale == (x * scale) - (zero_point * scale)
```

It emits two native DQ paths without a zero-point input:

1. `DequantizeLinear(x, scale)` computes `x * scale` for the even-length initializer.
2. The runtime scalar INT4 zero point is reshaped to `{1}`, duplicated by Concat to `{2}`, then `DequantizeLinear(duplicated_zero, scale)` computes `zero_point * scale`. Slice retains one FP32 value.
3. Subtract broadcasts that single FP32 value across the first result.

This avoids ever presenting the runtime value as TensorRT's special DQ zero-point input. It remains a genuine runtime engine input and may vary between executions.

The gate is limited to a static, rank-one, positive even-length INT4 initializer; runtime scalar FP32 scale; and runtime scalar INT4 zero point. Other shapes, types, initializer/runtime arrangements, and parameter ranks remain unchanged. A preprocessing test checks for the two DQ nodes and the Reshape/Concat/Slice/Sub structure.

### Artifact and tests

```text
Artifact: C:\Users\amadhavasrir\Downloads\bulding_files\test-env\onnxruntime_providers_nv_tensorrt_rtx.dq-step4-int4-runtime-zero.dll
SHA256: 98914EA23C371635D67BA6E01773955B8854FCF82162C7F32DFB67D867E66761
```

The target passed. Detailed logging showed one Myelin engine whose runtime signature is `x_scale (Float[]), x_zero_point (Int4[]) -> y (Float[1024])`, proving that the zero point remained a runtime input and the graph ran on TensorRT.

All seven DQ tests that remained after the earlier full-suite run now pass together:

```text
DequantizeLinearOpTest.Int4_LargeInitializerInput
DequantizeLinearOpTest.Int4NoZeroPoint
DequantizeLinearOpTest.Without_Zero_Point
DequantizeLinearOpTest.Per_Channel_Axis_Default
DequantizeLinearOpTest.Per_Channel_Axis_1_int8
DequantizeLinearContribOpTest.DequantizeLinear_1
DequantizeLinearContribOpTest.DequantizeLinear_2
```

That focused run contained zero `Invalid Node`, zero `No graph will run on TensorRT execution provider`, and zero serialized-engine build errors. A complete batched provider-suite run is still required to establish the final global failure count.

### Complete batched regression after Steps 1-4

Result directory:

```text
C:\Users\amadhavasrir\Downloads\bulding_files\results\dq-four-steps-full-20260817-164548
```

The runner enumerated 5,538 names. Five contain Google's `DISABLED_` marker and intentionally execute zero tests without `--gtest_also_run_disabled_tests`. Excluding those five, all 5,533 enabled tests produced a terminal result.

| Result | Before DQ follow-ups | After Steps 1-4 | Change |
|---|---:|---:|---:|
| Passed | 4,941 | 4,948 | +7 |
| Failed | 471 | 464 | -7 |
| Skipped | 121 | 121 | 0 |
| Enabled total | 5,533 | 5,533 | 0 |

Failure-set comparison found:

- newly introduced failures: **0**;
- previous failures now passing: **7**;
- genuine hangs or missing enabled tests: **0**.

The seven fixed names are exactly the seven DQ targets listed above. All 82 enabled test names containing `DequantizeLinear` now pass: **82 passed, 0 failed**.

As in the previous full run, the resume script wrote the five disabled names to `hangs.txt` because a zero-test invocation is interpreted as “no progress.” Their batch logs report zero executed tests; they are not process hangs.

Final provider-suite failure count after all four follow-up steps: **464**.

## Pad runtime-profile feasibility investigation

### Scope and failure inventory

No provider source behavior was changed in this investigation. The final 464-failure inventory contains 114 Pad tests
classified as dynamic shape/profile failures. They consist of 28 typed Pad scenarios repeated for the four datatypes
currently claimed by TensorRT RTX in this test set (112 tests), plus `PadOpTest.BoolType` and
`PadOpTest.ConstantPadAxes`.

The shared cause is that opset 11+ models supply `pads` as a runtime graph input. TensorRT classifies that input as a
shape tensor, while the current implicit-profile builder copies its container dimension (for example `[2]`) instead of
constructing bounds for its two runtime values.

### Provider-test controls

`PadOpTest/0.Pad_Constant_1D` failed without a profile and passed with both of the following explicit profiles:

```text
fixed:    min=[1,2], opt=[1,2], max=[1,2]
variable: min=[0,0], opt=[1,2], max=[2,3]
```

`PadOpTest/0.Pad_Constant_2D_negative_pads_1` also passed with a profile containing negative and positive values. A
deliberately over-broad minimum point caused TensorRT to warn that the profile was not self-consistent, demonstrating
that Pad bounds must describe valid combinations rather than merely large per-element ranges.

One shared nonnegative profile was also applied to four representative float tests covering every Pad mode exercised by
the failure set. `Pad_Spec_Example` (constant), `Pad_Edge_1D`, `Pad_Reflect_1D`, and `Pad_Wrap_1D` all passed: **4/4**.

### Same-engine runtime-value experiment

An isolated experiment under the untracked directory `build-phase1-ninja\pad-profile-experiment` created one Pad engine
with CPU fallback disabled and ran the same session repeatedly. With `min=[-1,-1]`, `opt=[0,0]`, and `max=[2,2]`, all
nine runtime pairs produced the expected result:

```text
[-1,-1], [-1,0], [0,-1], [-1,1], [1,-1],
[0,0], [1,1], [2,0], [0,2]
```

This proves that, unlike native CumSum's axis, Pad values do not have to be build-time constants. One TensorRT engine
can execute different positive and negative pad values when they remain inside a valid optimization profile.

### Current conclusion and implementation boundary

Runtime engine creation is not intrinsically required for Pad. The remaining problem is constructing sound implicit
bounds without seeing the runtime values. For each input dimension, pre-padding and post-padding are correlated because
their sum determines the output extent. A single rectangular min/max profile cannot represent every legal negative-pad
combination without also containing invalid points; TensorRT reports such invalid profile points during shape analysis.

Approximately 73 of the 114 failing Pad test names belong to scenarios whose observed test feeds are nonnegative and do
not use runtime `axes`. Static input/output shapes could produce bounded profiles that cover those observed feeds, but
doing so would implicitly exclude other ONNX-valid negative values that the same model could receive at runtime. This is
therefore a possible deliberately scoped policy/prototype, not yet a generally correct Pad implementation. No provider
code change has been made pending review of that semantic tradeoff.

## Slice runtime-profile feasibility investigation

### Scope and failure inventory

No provider source behavior was changed. The final 464-failure inventory contains 44 `SliceTest` failures. Slice opset
10+ supplies `starts`, `ends`, optional `axes`, and optional `steps` as graph inputs. In the ORT test helper, each normal
case is exercised first with these parameters as runtime inputs and then with them as initializers. The runtime form fails
during session initialization because the current implicit profile copies each parameter tensor's container dimension
(normally `[1]`) rather than supplying bounds for its values.

`SliceTest.Slice1D_Regular` was used as the provider-test control. It failed without explicit profiles and passed with the
exact runtime values `starts=[2]`, `ends=[4]`, and `axes=[0]`. Exact profiles also made the following representative tests
pass:

- `SliceTest.Slice2D_TwoAxes` with two-element `starts`, `ends`, and `axes`;
- `SliceTest.Slice1D_WithNegativeSteps_Regular` with negative bounds and `steps=[-1]`;
- `SliceTest.Slice1D_EndOutOfBounds` with `ends=[10]` for a six-element input;
- `SliceTest.Slice1D_InvalidStartEndRange`, which correctly produced an empty output;
- `SliceTest.OptionalAxesInputAloneMissing`, where `steps` is present but `axes` is omitted.

### Same-engine runtime-value matrix

An isolated runner and two small models were placed under the untracked directory
`build-phase1-ninja\slice-profile-experiment`. CPU fallback was disabled. Each variable-profile experiment created one
session and reused its single TensorRT engine for all listed runtime values. All 23 custom inference executions passed:

| Parameter under test | Values exercised in one engine | Result |
|---|---|---|
| `ends` | `2`, `4`, `6` with `starts=0` | 3/3 passed with output lengths 2, 4, and 6 |
| `starts` | `0`, `1`, `3` with `ends=6` | 3/3 passed with output lengths 6, 5, and 3 |
| `axes` | `0`, `1` on a rank-2 input | 6/6 passed across two fixed controls, a full-slice variable pair, and a distinct-output variable pair |
| positive `steps` | `1`, `2` | 2/2 variable-profile runs passed; both fixed controls also passed |
| negative `steps` | `-2`, `-1` | 2/2 variable-profile runs passed; both fixed controls also passed |
| negative bounds | `starts=-4`, `ends=-1` | passed |
| out-of-bounds bounds | `starts=-100`, `ends=100` | passed and clamped correctly |

The distinct-output axis control used the same session with `starts=1` and `ends=2`. Runtime `axis=0` produced shape
`[1,3]` and values `[3,4,5]`; runtime `axis=1` produced shape `[2,1]` and values `[1,4]`. This proves that the second axis
value was consumed at execution and was not merely ignored by an engine specialized for the first axis.

### Edge cases exposed after profiles were supplied

Two expected-failure tests remain separate error-propagation issues rather than runtime-profile restrictions:

- `SliceTest.InvalidAxesOutOfBounds`: TensorRT rejected axis 2 for a rank-2 tensor, but the EP returned only the generic
  serialized-engine failure instead of ORT's expected `axis outside of the tensor dimension count` text.
- `SliceTest.InvalidAxesDuplicates`: TensorRT detected the duplicate axis, but the EP again replaced the detailed error
  with the generic serialized-engine failure.

The two `SliceTest.EmptyDim` calls were subsequently isolated. The positive-step zero-extent call passed; only the
negative-step zero-extent call failed TensorRT engine construction (`starts operand cannot be negative at axis 0`).

### Conclusion for the runtime-engine count

Slice contributes **zero** tests to the set proven to require runtime engine creation. One prebuilt engine correctly
handled varying starts, ends, axes, and positive and negative steps when their values were covered by an explicit
optimization profile. A later exhaustive audit corrected the denominator to 46 failing Slice test instances and showed
that 42 can execute completely after diagnostic profiles (including sentinel-specialized profiles). The remaining four are one semantic mismatch, one
test containing a negative-step zero-extent limitation, and two validation/error-propagation cases.

### Exhaustive 46-test follow-up

The full per-input audit is recorded in [slice-failure-audit.md](slice-failure-audit.md). Its important additional
findings are:

- the 46 failing tests contain 50 distinct Slice input sets;
- 46/50 input sets execute natively after appropriate profiles or INT64-sentinel handling;
- a single profile can span positive and negative `steps`, so separate sign profiles are not required;
- the provider-option parser's `std::stoi` rejects literal INT64 profile values;
- raw runtime `INT64_MIN/MAX` values overflow under a wide profile, so profiles alone cannot handle sentinel cases;
- runtime values outside a shape-tensor profile can silently produce results specialized to the profile rather than the
  supplied value, making runtime profile-range validation a prerequisite for safe automatic profiles;
- `Slice1D_ReverseAllAxes_1` has an ORT/TensorRT semantic mismatch for negative step plus `INT32_MAX` end.

The resulting implementation order is: runtime range validation, ordinary Slice-aware profiles, runtime sentinel
normalization, an explicit policy for the reversal semantic mismatch, targeted zero-extent handling, and detailed axes
validation/error preservation.
