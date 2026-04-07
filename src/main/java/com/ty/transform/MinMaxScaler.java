package com.ty.transform;

import com.google.common.collect.Lists;

import java.util.Collections;
import java.util.List;

/**
 * 归一化处理
 *
 * <p>
 *      将数据点映射到了[0,1]区间
 * </p>
 *
 * <p>
 *     公式：Xstd = [X−X.min(axis=0)] / [X.max(axis=0) - X.min(axis=0)]
 * </p>
 *
 * <p>
 *     算法：
 *          第一步：求每列的最小值和最大值；
 *          第二步：求每个列中元素到最小值距离 占 该列最大值和最小值距离 的比例；
 * </p>
 *
 * @Author Tommy
 * @Date 2022/11/10
 */
public class MinMaxScaler {

    /**
     * 归一化
     *
     * @param data 数据集
     */
    public static void fit_transform(List<List<Double>> data) {
        // 列数
        int colSize = data.get(0).size();

        // 按列计算
        for (int j = 0; j < colSize; j++) {
            // 寻找每列的最大值与最小值
            double min = 0, max = 0;
            List<Double> colData = Lists.newArrayList(); // 存储列数据
            for (int i = 0; i < data.size(); i++) { // 按行循环
                colData.add(data.get(i).get(j)); // i x j 矩阵
            }
            Collections.sort(colData);
            min = colData.get(0);
            max = colData.get(colData.size() - 1);

            // 值归一化处理
            for (int i = 0; i < data.size(); i++) {
                data.get(i).set(j, normalize(data.get(i).get(j), min, max));
            }
        }
    }

    /**
     * 值归一化到[0,1]区间
     *
     * @param v     数值
     * @param min   数值所在列的最小值
     * @param max   数值所在列的最大值
     * @return 返回归一化后的值，在[0,1]区间
     */
    public static double normalize(double v, double min, double max) {
        return Math.round(((v - min) / (max - min)) * Math.pow(10, 8)) / Math.pow(10, 8);
    }
}
