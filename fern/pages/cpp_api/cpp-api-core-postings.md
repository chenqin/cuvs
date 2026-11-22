---
slug: api-reference/cpp-api-core-postings
---

# Postings

_Source header: `cuvs/core/postings.hpp`_

## Bitsets from postings lists

<a id="core-postings-run"></a>
### core::postings::run

One run of a postings list: the rows `lists[list][first, first + count)`.

```cpp
struct run {
  uint32_t list;
  int64_t first;
  int64_t count;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `list` | `uint32_t` | The postings list the run is in (an index into `evaluate`'s `lists`). |
| `first` | `int64_t` | Offset of the run's first row in the list. |
| `count` | `int64_t` | Rows in the run. |

<a id="core-postings-step-kind"></a>
### core::postings::step_kind

What a step of a program does.

```cpp
enum class step_kind : int {
  LEAF = 0,
  AND = 1,
  OR = 2,
  NOT = 3,
  ALL = 4,
  NONE = 5
};
```

**Values**

| Name | Value |
| --- | --- |
| `LEAF` | `0` |
| `AND` | `1` |
| `OR` | `2` |
| `NOT` | `3` |
| `ALL` | `4` |
| `NONE` | `5` |

<a id="core-postings-step"></a>
### core::postings::step

One step of a postfix program.

```cpp
struct step {
  int64_t first_run;
  int64_t n_runs;
  bool complement;
  int64_t exclude_run;
  step_kind kind;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `first_run` | `int64_t` | LEAF: its runs are `runs[first_run, first_run + n_runs)`. |
| `n_runs` | `int64_t` | LEAF: the number of its runs; 0 for a leaf of no rows. |
| `complement` | `bool` | LEAF: the rows in none of its runs instead. |
| `exclude_run` | `int64_t` | LEAF: when 0 or more, the rows of `runs[exclude_run]` are removed last, after `complement`. For example, a negated comparison that must not pass rows whose value is NULL is the complement of the compared values' runs with the NULL value's run excluded. |
| `kind` | [`step_kind`](/api-reference/cpp-api-core-postings#core-postings-step-kind) |  |

<a id="core-postings-workspace-size"></a>
### core::postings::workspace_size

Device scratch `evaluate` needs, in 32-bit words.

```cpp
auto workspace_size(raft::host_vector_view<const step, int64_t> program,
int64_t n_runs,
int64_t n_rows) -> int64_t;
```

One bitset of `n_rows` rows for each result the program holds at once beyond its first, and the uploaded runs.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `program` | in | [`raft::host_vector_view<const step, int64_t>`](/api-reference/cpp-api-core-postings#core-postings-step) | the steps, as for `evaluate` |
| `n_runs` | in | `int64_t` | the number of runs, as for `evaluate` |
| `n_rows` | in | `int64_t` | the rows of the bitset `evaluate` writes |

**Returns**

`int64_t`

the length of a `workspace` that `evaluate` takes

<a id="core-postings-evaluate"></a>
### core::postings::evaluate

Run a postfix program over postings lists into a bitset.

```cpp
void evaluate(
raft::resources const& res,
raft::host_vector_view<const step, int64_t> program,
raft::host_vector_view<const run, int64_t> runs,
raft::host_vector_view<const raft::device_vector_view<const uint32_t, int64_t>, int64_t> lists,
cuvs::core::bitset_view<uint32_t, int64_t> out,
int64_t first_row                                                    = 0,
std::optional<raft::device_vector_view<uint32_t, int64_t>> workspace = std::nullopt,
std::optional<raft::device_scalar_view<int64_t>> count               = std::nullopt);
```

Bit `i` of `out` is set when row `first_row + i` passes the program, for the rows `[first_row, first_row + out.size())`: the whole index, or the rows of one part of it, such as a shard. Rows of the lists outside that window are ignored; within a LEAF, rows that are in several of its runs are set once. The program must leave one result; an AND or OR needs two before it and a NOT one.

Everything runs on the stream of `res`, with no wait: `out`, `count` and `workspace` are ready for same-stream use. The runs are copied to the device before the call returns, so the host arrays can be released at once; the lists must stay unchanged until the stream reaches the work.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | RAFT resources: the stream, and the workspace memory when `workspace` is not given |
| `program` | in | [`raft::host_vector_view<const step, int64_t>`](/api-reference/cpp-api-core-postings#core-postings-step) | the steps, in postfix order |
| `runs` | in | [`raft::host_vector_view<const run, int64_t>`](/api-reference/cpp-api-core-postings#core-postings-run) | the runs the leaves name |
| `lists` | in | `raft::host_vector_view<const raft::device_vector_view<const uint32_t, int64_t>, int64_t>` | the postings lists, in device memory: row numbers of the index, below 2^32 |
| `out` | out | `cuvs::core::bitset_view<uint32_t, int64_t>` | the bitset of the rows `[first_row, first_row + out.size())`; every word is written |
| `first_row` | in | `int64_t` | the row bit 0 of `out` stands for<br />Default: `0`. |
| `workspace` | in | `std::optional<raft::device_vector_view<uint32_t, int64_t>>` | device scratch of at least `workspace_size(program, runs.size(), out.size())` words; allocated from the workspace memory of `res` when not given<br />Default: `std::nullopt`. |
| `count` | out | `std::optional<raft::device_scalar_view<int64_t>>` | when given, receives the number of rows that pass<br />Default: `std::nullopt`. |

**Returns**

`void`
