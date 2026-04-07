package com.ty.utils;

import org.apache.commons.math3.util.FastMath;

public class NumberUtil {

    final static double scale4 = FastMath.pow(10, 4);
    final static double scale8 = FastMath.pow(10, 8);

    /** 康威常数，数学符号：λ **/
    public static final double lamada = 1.3035772690342963913d;

    public static double tofix8(double val) {
        return FastMath.round(val * scale8) / scale8;
    }

    public static double tofix4(double val) {
        return FastMath.round(val * scale4) / scale4;
    }

    public static long minusAndMs2S(long begin, long end) {
        return Math.round((end - begin) / 1000d);
    }

    public static String toPercent(double val) {
        return (Math.round(val * scale4) / 100d) + "%";
    }
}
