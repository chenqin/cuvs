---
slug: api-reference/java-api-com-nvidia-cuvs-cagrasearchparams
---

# CagraSearchParams

_Java package: `com.nvidia.cuvs`_

```java
public class CagraSearchParams
```

CagraSearchParams encapsulates the logic for configuring and holding search
parameters.

## Public and Protected Members

### SINGLE_CTA

```java
SINGLE_CTA(0), /** * for small batch sizes */ MULTI_CTA(1), /** * MULTI_KERNEL */ MULTI_KERNEL(2), /** * AUTO */ AUTO(100)
```

for large batch sizes

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:38`_

### MULTI_CTA

```java
MULTI_CTA(1), /** * MULTI_KERNEL */ MULTI_KERNEL(2), /** * AUTO */ AUTO(100)
```

for small batch sizes

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:42`_

### MULTI_KERNEL

```java
MULTI_KERNEL(2), /** * AUTO */ AUTO(100)
```

MULTI_KERNEL

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:46`_

### AUTO

```java
AUTO(100)
```

AUTO

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:50`_

### HASH

```java
HASH(0), /** * SMALL */ SMALL(1), /** * AUTO_HASH */ AUTO_HASH(100)
```

HASH

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:69`_

### SMALL

```java
SMALL(1), /** * AUTO_HASH */ AUTO_HASH(100)
```

SMALL

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:73`_

### AUTO_HASH

```java
AUTO_HASH(100)
```

AUTO_HASH

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:77`_

### SCORE_ALL

```java
SCORE_ALL(0), /** * The distance is computed only for nodes that pass the filter, and only * these nodes enter the itopk list and are selected as parent nodes. The * child nodes of a parent node are its neighbors that pass the filter and, * for each neighbor that is filtered out (a bridge), the neighbors of the * bridge that pass the filter, up to the child capacity of the result * buffer (about the graph degree). The initial random seed nodes are random * nodes that pass the filter, each the nearest of up to as many such nodes * as the number of random samplings (see * {@link Builder#withNumRandomSamplings(int)}). The itopk size is not * increased according to the filtering rate, and the filtering rate of the * prefilter is not computed. */ PASSING_TWO_HOP(1)
```

The distance is computed for every neighbor of a parent node, whether or
not it passes the filter, and a filtered-out node can be selected as a
parent node. Filtered-out nodes are removed from the search results. The
MULTI_CTA algorithm increases the itopk size according to the filtering
rate to compensate for the filtered-out nodes in the itopk list.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:116`_

### PASSING_TWO_HOP

```java
PASSING_TWO_HOP(1)
```

The distance is computed only for nodes that pass the filter, and only
these nodes enter the itopk list and are selected as parent nodes. The
child nodes of a parent node are its neighbors that pass the filter and,
for each neighbor that is filtered out (a bridge), the neighbors of the
bridge that pass the filter, up to the child capacity of the result
buffer (about the graph degree). The initial random seed nodes are random
nodes that pass the filter, each the nearest of up to as many such nodes
as the number of random samplings (see
`Builder#withNumRandomSamplings(int)`). The itopk size is not
increased according to the filtering rate, and the filtering rate of the
prefilter is not computed.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:130`_

### getMaxQueries

```java
public int getMaxQueries()
```

Gets the maximum number of queries to search at the same time (batch size).

**Returns**

the maximum number of queries

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:210`_

### getITopKSize

```java
public int getITopKSize()
```

Gets the number of intermediate search results retained during the search.

**Returns**

the number of intermediate search results

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:219`_

### getMaxIterations

```java
public int getMaxIterations()
```

Gets the upper limit of search iterations.

**Returns**

the upper limit value

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:228`_

### getTeamSize

```java
public int getTeamSize()
```

Gets the number of threads used to calculate a single distance.

**Returns**

the number of threads configured

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:237`_

### getSearchWidth

```java
public int getSearchWidth()
```

Gets the number of graph nodes to select as the starting point for the search
in each iteration.

**Returns**

the number of graph nodes

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:247`_

### getMinIterations

```java
public int getMinIterations()
```

Gets the lower limit of search iterations.

**Returns**

the lower limit value

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:256`_

### getThreadBlockSize

```java
public int getThreadBlockSize()
```

Gets the thread block size.

**Returns**

the thread block size

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:265`_

### getHashmapMinBitlen

```java
public int getHashmapMinBitlen()
```

Gets the lower limit of hash map bit length.

**Returns**

the lower limit value

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:274`_

### getNumRandomSamplings

```java
public int getNumRandomSamplings()
```

Gets the number of iterations of initial random seed node selection.

**Returns**

the number of iterations

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:283`_

### getHashMapMaxFillRate

```java
public float getHashMapMaxFillRate()
```

Gets the upper limit of hash map fill rate.

**Returns**

the upper limit of hash map fill rate

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:292`_

### getRandXORMask

```java
public long getRandXORMask()
```

Gets the bit mask used for initial random seed node selection.

**Returns**

the bit mask value

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:301`_

### getCagraSearchAlgo

```java
public SearchAlgo getCagraSearchAlgo()
```

Gets which search implementation is configured.

**Returns**

the configured `SearchAlgo`

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:310`_

### getHashMapMode

```java
public HashMapMode getHashMapMode()
```

Gets the hash map mode configured.

**Returns**

the configured `HashMapMode`

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:319`_

### getFilteringRate

```java
public float getFilteringRate()
```

Gets the rate of nodes to be filtered-out, when filtering is used.

**Returns**

the filtering rate, or 0 when it is automatically calculated

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:328`_

### getFilterExpansion

```java
public FilterExpansion getFilterExpansion()
```

Gets how a filtered search traverses the graph.

**Returns**

the configured `FilterExpansion`

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:337`_

### getQueryBitsets

```java
public long getQueryBitsets()
```

Gets the device address of the array with one bitset pointer per query.

**Returns**

the device address, or 0 when there are no per-query bitsets

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:346`_

### Builder

```java
public Builder()
```

Default constructor.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:412`_

### withMaxQueries

```java
public Builder withMaxQueries(int maxQueries)
```

Sets the maximum number of queries to search at the same time (batch size).
Auto select when 0.

**Parameters**

| Name | Description |
| --- | --- |
| `maxQueries` | the maximum number of queries |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:421`_

### withItopkSize

```java
public Builder withItopkSize(int iTopKSize)
```

Sets the number of intermediate search results retained during the search.
This is the main knob to adjust trade off between accuracy and search speed.
Higher values improve the search accuracy.

**Parameters**

| Name | Description |
| --- | --- |
| `iTopKSize` | the number of intermediate search results |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:434`_

### withMaxIterations

```java
public Builder withMaxIterations(int maxIterations)
```

Sets the upper limit of search iterations. Auto select when 0.

**Parameters**

| Name | Description |
| --- | --- |
| `maxIterations` | the upper limit of search iterations |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:445`_

### withAlgo

```java
public Builder withAlgo(SearchAlgo cuvsCagraSearchAlgo)
```

Sets which search implementation to use.

**Parameters**

| Name | Description |
| --- | --- |
| `cuvsCagraSearchAlgo` | the `SearchAlgo` to use |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:456`_

### withTeamSize

```java
public Builder withTeamSize(int teamSize)
```

Sets the number of threads used to calculate a single distance. 4, 8, 16, or
32.

**Parameters**

| Name | Description |
| --- | --- |
| `teamSize` | the number of threads used to calculate a single distance |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:468`_

### withSearchWidth

```java
public Builder withSearchWidth(int searchWidth)
```

Sets the number of graph nodes to select as the starting point for the search
in each iteration.

**Parameters**

| Name | Description |
| --- | --- |
| `searchWidth` | the number of graph nodes to select |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:480`_

### withMinIterations

```java
public Builder withMinIterations(int minIterations)
```

Sets the lower limit of search iterations.

**Parameters**

| Name | Description |
| --- | --- |
| `minIterations` | the lower limit of search iterations |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:491`_

### withThreadBlockSize

```java
public Builder withThreadBlockSize(int threadBlockSize)
```

Sets the thread block size. 0, 64, 128, 256, 512, 1024. Auto selection when
0.

**Parameters**

| Name | Description |
| --- | --- |
| `threadBlockSize` | the thread block size |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:503`_

### withHashMapMode

```java
public Builder withHashMapMode(HashMapMode hashMapMode)
```

Sets the hash map type. Auto selection when AUTO.

**Parameters**

| Name | Description |
| --- | --- |
| `hashMapMode` | the `HashMapMode` |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:514`_

### withHashMapMinBitlen

```java
public Builder withHashMapMinBitlen(int hashMapMinBitlen)
```

Sets the lower limit of hash map bit length. More than 8.

**Parameters**

| Name | Description |
| --- | --- |
| `hashMapMinBitlen` | the lower limit of hash map bit length |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:525`_

### withHashMapMaxFillRate

```java
public Builder withHashMapMaxFillRate(float hashMapMaxFillRate)
```

Sets the upper limit of hash map fill rate. More than 0.1, less than 0.9.

**Parameters**

| Name | Description |
| --- | --- |
| `hashMapMaxFillRate` | the upper limit of hash map fill rate |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:536`_

### withNumRandomSamplings

```java
public Builder withNumRandomSamplings(int numRandomSamplings)
```

Sets the number of iterations of initial random seed node selection. 1 or
more.

**Parameters**

| Name | Description |
| --- | --- |
| `numRandomSamplings` | the number of iterations of initial random seed node selection |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:549`_

### withRandXorMask

```java
public Builder withRandXorMask(long randXORMask)
```

Sets the bit mask used for initial random seed node selection.

**Parameters**

| Name | Description |
| --- | --- |
| `randXORMask` | the bit mask used for initial random seed node selection |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:560`_

### withFilteringRate

```java
public Builder withFilteringRate(float filteringRate)
```

Sets the rate of nodes to be filtered-out, when filtering is used.

The value must be greater than 0.0 and less than 1.0. When set to 0
(default), the filtering rate is automatically calculated: each
`CagraIndex` search with a prefilter then counts the bits of the
prefilter and waits for the result, unless the filter expansion is
`FilterExpansion#PASSING_TWO_HOP`, which does not use the rate.
`MultiPartitionCagraSearch` then counts the bits of the partition
filters in every call.

**Parameters**

| Name | Description |
| --- | --- |
| `filteringRate` | the rate of nodes to be filtered-out |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:579`_

### withFilterExpansion

```java
public Builder withFilterExpansion(FilterExpansion filterExpansion)
```

Sets how a filtered search traverses the graph (see
`FilterExpansion`). Default: `FilterExpansion#SCORE_ALL`.

`FilterExpansion#PASSING_TWO_HOP` is implemented by the
MULTI_CTA algorithm only. It selects MULTI_CTA when the search algorithm
is `SearchAlgo#AUTO`. The other algorithms ignore it and log a
warning. `MultiPartitionCagraSearch` always uses
`FilterExpansion#SCORE_ALL`. The value has no effect on a search
without a prefilter.

**Parameters**

| Name | Description |
| --- | --- |
| `filterExpansion` | the `FilterExpansion` to use |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:598`_

### withQueryBitsets

```java
public Builder withQueryBitsets(long queryBitsets)
```

Sets per-query filter bitsets for MULTI_CTA search. Not used when 0
(default).

When not 0, this is the device address of an array with one pointer
per query of the search call. Each pointer is the device address of the
32-bit words of the bitset for that query, with one bit per row of the
index: row `i` passes the filter when bit `i % 32` of word
`i / 32` is set. A null pointer leaves that query unfiltered. The
bitset of the prefilter given to the query is not read, so queries with
different filters can share one search call.

The query must be given a prefilter with
`CagraQuery.Builder#withPrefilter`. The search algorithm must be
`SearchAlgo#MULTI_CTA` or `SearchAlgo#AUTO`, which then
selects MULTI_CTA; the search fails with any other algorithm. With
`FilterExpansion#SCORE_ALL`, the filtering rate must be greater
than 0, because it is not calculated from per-query bitsets.
`FilterExpansion#PASSING_TWO_HOP` does not use it.
`MultiPartitionCagraSearch` and `TieredIndex` do not support
per-query bitsets and fail when this value is not 0.

The array and the bitsets are not copied: they must reside on the
device that the search runs on and must remain valid until the work that
the search submits to the CUDA stream has completed.

**Parameters**

| Name | Description |
| --- | --- |
| `queryBitsets` | the device address of an array with one bitset pointer per query, or 0 for none |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:633`_

### build

```java
public CagraSearchParams build()
```

Builds an instance of `CagraSearchParams` with passed search
parameters.

**Returns**

an instance of CagraSearchParams

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:644`_

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:13`_
