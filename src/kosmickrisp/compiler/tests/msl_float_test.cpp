/*
 * Copyright 2026 Gustavo Noronha Silva
 * SPDX-License-Identifier: MIT
 */

#include <gtest/gtest.h>
#include <locale.h>

#include "msl_float.h"

/* The driver runs inside the application, under whatever locale it adopted. Firefox adopts the
 * user's, and under a comma-decimal LC_NUMERIC (pt_BR) every float constant came out as
 * `float(-5,00000000000000000e-01)`, which Metal reads as two initialisers and refuses: no
 * pipeline built, nothing drawn. */
TEST(msl_float, a_literal_ignores_a_comma_decimal_locale)
{
   locale_t comma = newlocale(LC_NUMERIC_MASK, "pt_BR.UTF-8", (locale_t)0);
   if (!comma)
      GTEST_SKIP() << "no pt_BR.UTF-8 locale to test under";
   locale_t was = uselocale(comma);
   char buf[64];
   msl_format_float(buf, sizeof(buf), -0.5);
   uselocale(was);
   freelocale(comma);
   EXPECT_STREQ(buf, "-5.00000000000000000e-01");
}

/* The same digits as before under the C locale: the MSL a shader compiles to does not change. */
TEST(msl_float, a_literal_is_exact)
{
   char buf[64];
   msl_format_float(buf, sizeof(buf), 0.1f);
   EXPECT_STREQ(buf, "1.00000001490116119e-01");
}
