/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import com.nvidia.cuvs.CagraSearchParams.FilterExpansion;
import org.junit.Test;

public class CagraSearchParamsTest {

  @Test
  public void filterParamsDefaultToZero() {
    CagraSearchParams params = new CagraSearchParams.Builder().build();

    assertEquals(0.0f, params.getFilteringRate(), 0.0f);
    assertEquals(FilterExpansion.SCORE_ALL, params.getFilterExpansion());
    assertEquals(0L, params.getQueryBitsets());
  }

  @Test
  public void filterParamsAreKept() {
    long queryBitsets = 0x7f0012345600L;
    CagraSearchParams params =
        new CagraSearchParams.Builder()
            .withFilteringRate(0.9f)
            .withFilterExpansion(FilterExpansion.PASSING_TWO_HOP)
            .withQueryBitsets(queryBitsets)
            .build();

    assertEquals(0.9f, params.getFilteringRate(), 0.0f);
    assertEquals(FilterExpansion.PASSING_TWO_HOP, params.getFilterExpansion());
    assertEquals(queryBitsets, params.getQueryBitsets());

    String text = params.toString();
    assertTrue(text, text.contains("filteringRate=0.9"));
    assertTrue(text, text.contains("filterExpansion=PASSING_TWO_HOP"));
    assertTrue(text, text.contains("queryBitsets=" + queryBitsets));
  }

  @Test
  public void filterExpansionValuesMatchTheCEnum() {
    assertEquals(0, FilterExpansion.SCORE_ALL.value);
    assertEquals(1, FilterExpansion.PASSING_TWO_HOP.value);
  }
}
