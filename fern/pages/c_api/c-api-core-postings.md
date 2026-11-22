---
slug: api-reference/c-api-core-postings
---

# Postings

_Source header: `cuvs/core/postings.h`_

## C API for bitsets from postings lists

<a id="cuvspostingsstepkind"></a>
### cuvsPostingsStepKind

What a step of a postings program does.

```c
typedef enum {
  CUVS_POSTINGS_LEAF = 0,
  CUVS_POSTINGS_AND = 1,
  CUVS_POSTINGS_OR = 2,
  CUVS_POSTINGS_NOT = 3,
  CUVS_POSTINGS_ALL = 4,
  CUVS_POSTINGS_NONE = 5
} cuvsPostingsStepKind;
```

**Values**

| Name | Value |
| --- | --- |
| `CUVS_POSTINGS_LEAF` | `0` |
| `CUVS_POSTINGS_AND` | `1` |
| `CUVS_POSTINGS_OR` | `2` |
| `CUVS_POSTINGS_NOT` | `3` |
| `CUVS_POSTINGS_ALL` | `4` |
| `CUVS_POSTINGS_NONE` | `5` |

<a id="cuvspostingsrun"></a>
### cuvsPostingsRun

One run of a postings list: the rows `lists[list][first, first + count)`.

```c
typedef struct {
  uint32_t list;
  int64_t first;
  int64_t count;
} cuvsPostingsRun;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `list` | `uint32_t` |  |
| `first` | `int64_t` |  |
| `count` | `int64_t` |  |

<a id="cuvspostingsstep"></a>
### cuvsPostingsStep

One step of a postfix program; the fields after `kind` are for leaves.

```c
typedef struct {
  int64_t first_run;
  int32_t complement;
  int64_t exclude_run;
  cuvsPostingsStepKind kind;
  int64_t n_runs;
} cuvsPostingsStep;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `first_run` | `int64_t` | The leaf's runs are `runs[first_run, first_run + n_runs)`. |
| `complement` | `int32_t` | Nonzero: the rows in none of its runs instead. |
| `exclude_run` | `int64_t` | When 0 or more, the rows of `runs[exclude_run]` are removed last; -1 for none. |
| `kind` | [`cuvsPostingsStepKind`](/api-reference/c-api-core-postings#cuvspostingsstepkind) |  |
| `n_runs` | `int64_t` |  |

<a id="cuvspostingsworkspacesize"></a>
### cuvsPostingsWorkspaceSize

Device scratch cuvsPostingsEvaluate needs, in 32-bit words.

```c
cuvsError_t cuvsPostingsWorkspaceSize(const cuvsPostingsStep *steps,
int64_t n_steps,
int64_t n_runs,
int64_t n_rows,
int64_t *words);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `steps` | in | [`const cuvsPostingsStep*`](/api-reference/c-api-core-postings#cuvspostingsstep) | the program |
| `n_steps` | in | `int64_t` | its length |
| `n_runs` | in | `int64_t` | the number of runs |
| `n_rows` | in | `int64_t` | the rows of the bitset written |
| `words` | out | `int64_t*` | the length of a workspace cuvsPostingsEvaluate takes |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvspostingsevaluate"></a>
### cuvsPostingsEvaluate

Run a postfix program over postings lists into a bitset, on the GPU.

```c
cuvsError_t cuvsPostingsEvaluate(
cuvsResources_t res, const cuvsPostingsStep *steps, int64_t n_steps,
const cuvsPostingsRun *runs, int64_t n_runs, DLManagedTensor *const *lists,
uint32_t n_lists, DLManagedTensor *bitset, int64_t n_rows,
int64_t first_row, DLManagedTensor *workspace, DLManagedTensor *count);
```

Bit `i` of `bitset` is set when row `first_row + i` passes, for the rows `[first_row, first_row + n_rows)`; rows of the lists outside them are ignored. Everything runs on the stream of `res` with no wait; the steps and runs are copied before the call returns. See cuvs::core::postings::evaluate.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `steps` | in | [`const cuvsPostingsStep*`](/api-reference/c-api-core-postings#cuvspostingsstep) | the program, in postfix order |
| `n_steps` | in | `int64_t` | its length |
| `runs` | in | [`const cuvsPostingsRun*`](/api-reference/c-api-core-postings#cuvspostingsrun) | the runs the leaves name |
| `n_runs` | in | `int64_t` | their number |
| `lists` | in | `DLManagedTensor *const*` | `n_lists` 1-D uint32 device tensors: the postings lists (row numbers) |
| `n_lists` | in | `uint32_t` | their number |
| `bitset` | out | `DLManagedTensor*` | 1-D uint32 device tensor of at least (n_rows + 31) / 32 words |
| `n_rows` | in | `int64_t` | the rows of the bitset |
| `first_row` | in | `int64_t` | the row bit 0 stands for |
| `workspace` | in | `DLManagedTensor*` | 1-D uint32 device tensor of at least cuvsPostingsWorkspaceSize words, 8-byte aligned, or NULL to allocate from the workspace memory of `res` |
| `count` | out | `DLManagedTensor*` | 1-D int64 device tensor of one element receiving the rows that pass, or NULL |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)
