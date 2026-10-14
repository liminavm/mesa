/*
 * Copyright 2026 Gustavo Noronha Silva
 * SPDX-License-Identifier: MIT
 */

#include "msl_float.h"

#include <float.h>
#include <locale.h>
#include <stdio.h>
#ifdef __APPLE__
#include <xlocale.h>
#endif

#include "util/u_call_once.h"

static locale_t c_numeric;

static void
make_c_numeric(void)
{
   c_numeric = newlocale(LC_NUMERIC_MASK, "C", (locale_t)0);
}

void
msl_format_float(char *buf, size_t size, double v)
{
   static util_once_flag once = UTIL_ONCE_FLAG_INIT;
   util_call_once(&once, make_c_numeric);
   /* uselocale is this thread's alone, so the application's other threads keep theirs. */
   locale_t was = c_numeric ? uselocale(c_numeric) : (locale_t)0;
   snprintf(buf, size, "%.*le", DBL_DECIMAL_DIG, v);
   if (c_numeric)
      uselocale(was);
}
