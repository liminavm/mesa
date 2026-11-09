/*
 * Copyright 2026 Gustavo Noronha Silva
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The digits of an MSL floating literal for v, exact for a double ("%.*le", DBL_DECIMAL_DIG). The
 * text is source for the Metal compiler, so its radix character is '.' whatever LC_NUMERIC the
 * process using the driver has adopted. */
void msl_format_float(char *buf, size_t size, double v);

#ifdef __cplusplus
}
#endif
