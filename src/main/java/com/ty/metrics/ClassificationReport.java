package com.ty.metrics;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.ty.utils.NumberUtil;
import org.apache.commons.math3.util.FastMath;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collector;
import java.util.stream.Collectors;

/**
 * 模型评价指标
 *
 * @Author Tommy
 * @Date 2022/11/13
 */
public class ClassificationReport {

    private List<Double> y_test;
    private List<Integer> predict_result;
    private List<Double> pre_reslut_proba;

    /** 指标：MAPE（平均绝对百分比误差） **/
    private double mape;

    /** 指标：RMSE（均方根误差） **/
    private double rmse;

    /** TP、TN、FP、FN **/
    private int tp, tn, fp, fn;

    public ClassificationReport(List<Double> y_test, List<Integer> predict_result, List<Double> pre_reslut_proba) {
        this.y_test = y_test;
        this.predict_result = predict_result;
        this.pre_reslut_proba = pre_reslut_proba;

        this.statistics();
    }

    /**
     * 统计模型指标
     */
    public void statistics() {
        this.generalMetrics();
        this.MAPE();
        this.RMSE();
    }

    /**
     * 一般性信息：TP、TN、FP、FN
     */
    public void generalMetrics() {
        for (int i = 0; i < y_test.size(); i++) {
            double y = y_test.get(i);
            double y_ = predict_result.get(i);

            if (y == y_) {
                if (y_ == 1) {
                    this.tp++; // 真正
                } else {
                    this.tn++; // 真负
                }
            } else {
                if (y_ == 1) {
                    this.fp++; // 假正（误报）
                } else {
                    this.fn++; // 假负（漏报）
                }
            }
        }
    }

    /**
     * MAPE（平均绝对百分比误差）
     */
    public double MAPE() {
        int n = y_test.size();
        double sum  = 0;
        for (int i = 0; i < n; i++) {
            double y = y_test.get(i);
            if (0 != y) {
                sum += FastMath.abs((y - pre_reslut_proba.get(i)) / y);
            }
        }
        return this.mape = sum / n;
    }

    /**
     * RMSE（均方根误差）
     */
    public double RMSE() {
        int n = y_test.size();
        double sum  = 0;
        for (int i = 0; i < n; i++) {
            double y = y_test.get(i);
            sum += FastMath.pow(y - pre_reslut_proba.get(i), 2);
        }
        return this.rmse = FastMath.sqrt(sum / n);
    }

    /**
     * 准确率
     *
     * @return double
     */
    public double ACC() {
        return NumberUtil.tofix4((this.tp + this.tn) / (this.tp + this.tn + this.fp + this.fn * 1d));
    }

    /**
     * 精确率
     *
     * @return double
     */
    public double precision() {
        return NumberUtil.tofix4(this.tp / (this.tp + this.fp * 1d));
    }

    /**
     * 召回率
     *
     * @return double
     */
    public double recall() {
        return NumberUtil.tofix4(this.tp / (this.tp + this.fn * 1d));
    }

    /**
     * F1值
     *
     * @return double
     */
    public double f1_score() {
        return NumberUtil.tofix4(2d * this.tp / (2 * this.tp + this.fp + this.fn));
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();

        sb.append("TP = " + this.tp + "\n");
        sb.append("TN = " + this.tn + "\n");
        sb.append("FP(误报) = " + this.fp + "\n");
        sb.append("FN(漏报) = " + this.fn + "\n");
        sb.append("\n");

        sb.append("ACC(准确率)：\t" + NumberUtil.toPercent(this.ACC()) + "\n");
        sb.append("P(精确率)：\t\t" + NumberUtil.toPercent(this.precision()) + "\n");
        sb.append("Recall(召回率)：\t" + NumberUtil.toPercent(this.recall()) + "\n");
        sb.append("F1：\t" + this.f1_score() + "\n");

        sb.append("\n");

        sb.append("RMSE(均方根误差)：\t" + this.rmse + "\n");
        sb.append("MAPE(平均绝对百分比误差)：\t" + NumberUtil.toPercent(this.mape) + "\n");
        return sb.toString();
    }
}
