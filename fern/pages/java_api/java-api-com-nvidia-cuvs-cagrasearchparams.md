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

## Public Members

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
SCORE_ALL(0), /** * Only nodes that pass the filter are scored, kept and expanded. A rejected * neighbor is not scored: its own neighbors are examined instead. The search * starts from random nodes that pass the filter (as many per start slot as * the number of random samplings, the nearest kept). For filters that reject * most of the index. */ PASSING_TWO_HOP(1)
```

Every neighbor of an expanded node is scored and may be expanded in turn,
whether or not it passes the filter; rejected nodes are dropped from the
results.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:99`_

### PASSING_TWO_HOP

```java
PASSING_TWO_HOP(1)
```

Only nodes that pass the filter are scored, kept and expanded. A rejected
neighbor is not scored: its own neighbors are examined instead. The search
starts from random nodes that pass the filter (as many per start slot as
the number of random samplings, the nearest kept). For filters that reject
most of the index.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:107`_

### getMaxQueries

```java
public int getMaxQueries()
```

Gets the maximum number of queries to search at the same time (batch size).

**Returns**

the maximum number of queries

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:187`_

### getITopKSize

```java
public int getITopKSize()
```

Gets the number of intermediate search results retained during the search.

**Returns**

the number of intermediate search results

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:196`_

### getMaxIterations

```java
public int getMaxIterations()
```

Gets the upper limit of search iterations.

**Returns**

the upper limit value

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:205`_

### getTeamSize

```java
public int getTeamSize()
```

Gets the number of threads used to calculate a single distance.

**Returns**

the number of threads configured

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:214`_

### getSearchWidth

```java
public int getSearchWidth()
```

Gets the number of graph nodes to select as the starting point for the search
in each iteration.

**Returns**

the number of graph nodes

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:224`_

### getMinIterations

```java
public int getMinIterations()
```

Gets the lower limit of search iterations.

**Returns**

the lower limit value

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:233`_

### getThreadBlockSize

```java
public int getThreadBlockSize()
```

Gets the thread block size.

**Returns**

the thread block size

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:242`_

### getHashmapMinBitlen

```java
public int getHashmapMinBitlen()
```

Gets the lower limit of hash map bit length.

**Returns**

the lower limit value

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:251`_

### getNumRandomSamplings

```java
public int getNumRandomSamplings()
```

Gets the number of iterations of initial random seed node selection.

**Returns**

the number of iterations

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:260`_

### getHashMapMaxFillRate

```java
public float getHashMapMaxFillRate()
```

Gets the upper limit of hash map fill rate.

**Returns**

the upper limit of hash map fill rate

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:269`_

### getRandXORMask

```java
public long getRandXORMask()
```

Gets the bit mask used for initial random seed node selection.

**Returns**

the bit mask value

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:278`_

### getCagraSearchAlgo

```java
public SearchAlgo getCagraSearchAlgo()
```

Gets which search implementation is configured.

**Returns**

the configured `SearchAlgo`

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:287`_

### getHashMapMode

```java
public HashMapMode getHashMapMode()
```

Gets the hash map mode configured.

**Returns**

the configured `HashMapMode`

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:296`_

### getFilteringRate

```java
public float getFilteringRate()
```

Gets the fraction of the index's rows that the filter removes.

**Returns**

the filtering rate, or 0 when each search computes it from the prefilter

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:306`_

### getFilterExpansion

```java
public FilterExpansion getFilterExpansion()
```

Gets how a filtered MULTI_CTA search walks the graph.

**Returns**

the configured `FilterExpansion`

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:315`_

### getQueryBitsets

```java
public long getQueryBitsets()
```

Gets the device address of the array with one bitset pointer per query.

**Returns**

the device address, or 0 when there are no per-query bitsets

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:324`_

### Builder

```java
public Builder()
```

Default constructor.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:390`_

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

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:399`_

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

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:412`_

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

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:423`_

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

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:434`_

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

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:446`_

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

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:458`_

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

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:469`_

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

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:481`_

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

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:492`_

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

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:503`_

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

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:514`_

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

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:527`_

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

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:538`_

### withFilteringRate

```java
public Builder withFilteringRate(float filteringRate)
```

Sets the fraction of the index's rows that the filter removes. More than
0, less than 1. When 0 (the default), it is computed from the prefilter by
each search that needs it, which reads the whole bitset and waits for the
result.

**Parameters**

| Name | Description |
| --- | --- |
| `filteringRate` | the fraction of the index's rows that the filter removes |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:553`_

### withFilterExpansion

```java
public Builder withFilterExpansion(FilterExpansion filterExpansion)
```

Sets how a filtered MULTI_CTA search walks the graph. With
`FilterExpansion#PASSING_TWO_HOP`, `SearchAlgo#AUTO` selects
MULTI_CTA. Other algorithms ignore it, and so does
`MultiPartitionCagraSearch`.

**Parameters**

| Name | Description |
| --- | --- |
| `filterExpansion` | the `FilterExpansion` to use |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:567`_

### withQueryBitsets

```java
public Builder withQueryBitsets(long queryBitsets)
```

Sets one bitset per query for a filtered MULTI_CTA search. No per-query
bitsets when 0 (the default). Otherwise the device address of an array
with one pointer per query, each to the 32-bit words of that query's own
bitset over the index's rows (bit `i % 32` of word `i / 32` is
set when row `i` passes), on the device being searched. A null
pointer leaves that query unfiltered.

The query must still be given a prefilter with
`CagraQuery.Builder#withPrefilter`. Its bitset is then not read, so
queries with different filters can share one search call. The filtering
rate must be set unless the filter expansion is
`FilterExpansion#PASSING_TWO_HOP`, and the search algorithm must be
`SearchAlgo#MULTI_CTA` or `SearchAlgo#AUTO`, which then selects
MULTI_CTA.
`MultiPartitionCagraSearch` does not take per-query bitsets.

The array and the bitsets are not copied: they must stay valid until the
search's work on the CUDA stream has finished.

**Parameters**

| Name | Description |
| --- | --- |
| `queryBitsets` | the device address of an array with one bitset pointer per query, or 0 for none |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:596`_

### build

```java
public CagraSearchParams build()
```

Builds an instance of `CagraSearchParams` with passed search
parameters.

**Returns**

an instance of CagraSearchParams

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:607`_

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:13`_
